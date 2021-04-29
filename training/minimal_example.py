import torch
from tqdm import tqdm
import numpy as np
import nibabel as nib
import collections
from collections import OrderedDict

import catalyst
import pandas as pd

from catalyst.contrib.utils.pandas import dataframe_to_list
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
from catalyst.data import ReaderCompose
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from catalyst.callbacks import CheckpointCallback
from torch.nn import functional as F
from typing import List

from brain_dataset import BrainDataset
from reader import NiftiFixedVolumeReader, NiftiReader
from model import MeshNet, UNet
from catalyst import metrics
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.dl import Runner, LRFinder

from catalyst.metrics.functional._segmentation import dice
from catalyst.contrib.nn.criterion.focal import FocalLossMultiClass


def get_loaders(
    random_state: int,
    volume_shape: List[int],
    subvolume_shape: List[int],
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    batch_size: int = 16,
    num_workers: int = 10,
) -> dict:

    datasets = {}
    open_fn = ReaderCompose(
        [
            NiftiFixedVolumeReader(input_key="images", output_key="images"),
            NiftiReader(input_key="nii_labels", output_key="targets"),

        ]
    )

    for mode, source in zip(("train", "validation", "infer"),
                            (in_csv_train, in_csv_valid, in_csv_infer)):
        if mode == "infer":
            n_subvolumes = 512
        else:
            n_subvolumes = 128

        if source is not None and len(source) > 0:
            dataset = BrainDataset(
                list_data=dataframe_to_list(pd.read_csv(source)),
                list_shape=volume_shape,
                list_sub_shape=subvolume_shape,
                open_fn=open_fn,
                n_subvolumes=n_subvolumes,
                mode=mode,
                input_key="images",
                output_key="targets",
            )

        datasets[mode] = {"dataset": dataset}

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    train_loader = DataLoader(dataset=datasets['train']['dataset'], batch_size=batch_size,
                              shuffle=True, worker_init_fn=worker_init_fn,
                              num_workers=16, pin_memory=True)
    valid_loader = DataLoader(dataset=datasets['validation']['dataset'],
                              shuffle=True, worker_init_fn=worker_init_fn,
                              batch_size=batch_size,
                              num_workers=16, pin_memory=True,drop_last=True)
    test_loader = DataLoader(dataset=datasets['infer']['dataset'],
                             batch_size=batch_size, worker_init_fn=worker_init_fn,
                             num_workers=16, pin_memory=True,drop_last=True)
    train_loaders = collections.OrderedDict()
    infer_loaders = collections.OrderedDict()
    train_loaders["train"] = BatchPrefetchLoaderWrapper(train_loader)
    train_loaders["valid"] = BatchPrefetchLoaderWrapper(valid_loader)
    infer_loaders['infer'] = BatchPrefetchLoaderWrapper(test_loader)

    return train_loaders, infer_loaders

volume_shape = [256, 256, 256]
subvolume_shape = [38, 38, 38]
train_loaders, infer_loaders = get_loaders(0, volume_shape, subvolume_shape,
                                           "./data/dataset_train.csv",
                                           "./data/dataset_valid.csv",
                                           "./data/dataset_infer.csv", )
                                           )


class CustomRunner(Runner):

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def predict_batch(self, batch):
        # model inference step
        batch = batch[0]
        return self.model(batch['images'].float().to(self.device)), batch['coords']

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "macro_dice"]
        }

    def handle_batch(self, batch):

        # model train/valid step
        batch = batch[0]
        x, y = batch['images'].float(), batch['targets']

        if self.is_train_loader:
            self.optimizer.zero_grad()

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        one_hot_targets = (
            torch.nn.functional.one_hot(y, 31)
            .permute(0, 4, 1, 2, 3)
            .cuda()
            )

        logits_softmax = F.softmax(y_hat)
        macro_dice = dice(logits_softmax, one_hot_targets, mode='macro')

        self.batch_metrics.update({"loss": loss,
                                   'macro_dice': macro_dice})

        for key in ["loss", "macro_dice"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        for key in ["loss", "macro_dice"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


n_classes = 31
n_epochs = 30
meshnet = MeshNet(n_channels=1, n_classes=n_classes)

logdir = "logs/meshnet_mindboggle"

optimizer = torch.optim.Adam(meshnet.parameters(), lr=0.02)


scheduler = OneCycleLR(optimizer, max_lr=.02,
                       epochs=n_epochs, steps_per_epoch=len(train_loaders['train']))

#scheduler = LRFinder(final_lr=1.0)

runner = CustomRunner()
runner.train(model=meshnet, optimizer=optimizer, loaders=train_loaders,
             num_epochs=n_epochs, scheduler=scheduler,
             callbacks=[CheckpointCallback(logdir=logdir)], logdir=logdir, verbose=True)

segmentations = {}
for subject in range(infer_loaders['infer'].dataset.subjects):
    segmentations[subject] = torch.zeros(tuple(np.insert(volume_shape, 0, n_classes)), dtype=torch.uint8)


def voxel_majority_predict_from_subvolumes(loader, n_classes, segmentations):
    if segmentations is None:
        for subject in range(loader.dataset.subjects):
            segmentations[subject] = torch.zeros(
                tuple(np.insert(loader.volume_shape, 0, n_classes)),
                dtype=torch.uint8).cpu()

    prediction_n = 0
    for inference in tqdm(runner.predict_loader(loader=loader)):
        coords = inference[1].cpu()
        _, predicted = torch.max(F.log_softmax(inference[0].cpu(), dim=1), 1)
        for j in range(predicted.shape[0]):
            c_j = coords[j][0]
            subj_id = prediction_n // loader.dataset.n_subvolumes
            for c in range(n_classes):
                segmentations[subj_id][c, c_j[0, 0]:c_j[0, 1],
                                       c_j[1, 0]:c_j[1, 1],
                                       c_j[2, 0]:c_j[2, 1]] += (predicted[j] == c)
            prediction_n += 1

    for i in segmentations.keys():
        segmentations[i] = torch.max(segmentations[i], 0)[1]
    return segmentations

segmentations = voxel_majority_predict_from_subvolumes(infer_loaders['infer'],
                                                       n_classes, segmentations)
subject_metrics = []
for subject, subject_data in enumerate(tqdm(infer_loaders['infer'].dataset.data)):
    seg_labels = nib.load(subject_data['nii_labels']).get_fdata()
    segmentation_labels = torch.nn.functional.one_hot(
        torch.from_numpy(seg_labels).to(torch.int64), n_classes)

    inference_dice = dice(
        torch.nn.functional.one_hot(
            segmentations[subject], n_classes).permute(0, 3, 1, 2),
        segmentation_labels.permute(0, 3, 1, 2)).detach().numpy()
    macro_inference_dice = dice(
        torch.nn.functional.one_hot(segmentations[subject], n_classes).permute(0, 3, 1, 2),
        segmentation_labels.permute(0, 3, 1, 2), mode='macro').detach().numpy()
    subject_metrics.append((inference_dice, macro_inference_dice))

per_class_df = pd.DataFrame([metric[0] for metric in subject_metrics])
macro_df = pd.DataFrame([metric[1] for metric in subject_metrics])
print(per_class_df, macro_df)
print(macro_df.mean())

