import torch
import collections
from collections import OrderedDict

import catalyst
import pandas as pd

from catalyst.contrib.utils.pandas import dataframe_to_list
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from catalyst.data import ReaderCompose
from torch.optim.lr_scheduler import CosineAnnealingLR
from catalyst.callbacks import CheckpointCallback
from torch.nn import functional as F
from typing import List

from brain_dataset import BrainDataset
from reader import NiftiReader_Image, NiftiReader_Mask
from model import MeshNet
from custom_metrics import custom_dice_metric
from catalyst import metrics
from catalyst.data import BatchPrefetchLoaderWrapper


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
            NiftiReader_Image(input_key="images", output_key="images"),
            NiftiReader_Mask(input_key="nii_labels", output_key="targets"),
        ]
    )

    for mode, source in zip(("train", "validation", "infer"),
                            (in_csv_train, in_csv_valid, in_csv_infer)):
        if source is not None and len(source) > 0:
            dataset = BrainDataset(
                shared_dict={},
                list_data=dataframe_to_list(pd.read_csv(source)),
                list_shape=volume_shape,
                list_sub_shape=subvolume_shape,
                open_fn=open_fn,
                n_subvolumes=128,
                mode=mode,
                input_key="images",
                output_key="targets",
            )

        if mode in ["train", "validation"]:
            sampler = RandomSampler(
                data_source=dataset, replacement=True,
                num_samples=len(pd.read_csv(source)) * 128)
        else:
            sampler = SequentialSampler(data_source=dataset)

        datasets[mode] = {"dataset": dataset, "sampler": sampler}

    train_loader = DataLoader(dataset=datasets['train']['dataset'], batch_size=batch_size,
                              sampler=datasets['train']['sampler'],
                              num_workers=20, pin_memory=True)
    valid_loader = DataLoader(dataset=datasets['validation']['dataset'],
                              batch_size=batch_size,
                              sampler=datasets['validation']['sampler'],
                              num_workers=20, pin_memory=True,drop_last=True)
    test_loader = DataLoader(dataset=datasets['infer']['dataset'],
                             batch_size=1, sampler=datasets['infer']['sampler'],
                             num_workers=20, pin_memory=True,drop_last=True)
    train_loaders = collections.OrderedDict()
    infer_loaders = collections.OrderedDict()
    train_loaders["train"] = BatchPrefetchLoaderWrapper(train_loader)
    train_loaders["valid"] = BatchPrefetchLoaderWrapper(valid_loader)
    infer_loaders['infer'] = BatchPrefetchLoaderWrapper(test_loader)

    return train_loaders, infer_loaders

train_loaders, infer_loaders = get_loaders(0, [256, 256, 256], [38, 38, 38],
                                           "data/dataset_train.csv", "data/dataset_valid.csv", "data/dataset_infer.csv", )


class CustomRunner(catalyst.dl.Runner):

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def predict_batch(self, batch):
        # model inference step
        return self.model(batch['images'].to(self.device)), batch['coords']

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "dice"]
        }

    def handle_batch(self, batch):
        # model train/valid step
        x, y = batch['images'], batch['targets']
        y_hat = self.model(x)

        loss = F.cross_entropy(y_hat, y)
        self.batch_metrics.update({"loss": loss, "dice": custom_dice_metric(y_hat.float(), y, num_classes=60)})

        for key in ["loss", "dice"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner):
        for key in ["dice", "loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


n_classes = 60
torch.backends.cudnn.deterministic = False
meshnet = MeshNet(n_channels=1, n_classes=n_classes)

logdir = "logs/meshnet"

optimizer = torch.optim.Adam(meshnet.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=30)

runner = CustomRunner()
runner.train(model=meshnet, optimizer=optimizer, loaders=train_loaders, num_epochs=1, scheduler=scheduler,
             callbacks=[CheckpointCallback(logdir=logdir)], logdir=logdir, verbose=True)


def voxel_majority_predict_from_subvolumes(loader, volume_shape, n_classes):
    segmentations = {}
    for subject in range(loader.dataset.subjects):
        segmentations[subject] = torch.zeros(tuple(np.insert(volume_shape, 0, n_classes)), dtype=torch.uint8)

    for inference in runner.predict_loader(loader=loader):
        subj_id = loader.dataset.subjects // len(loader.dataset.coords)
        coords = inference[1]
        _, predicted = torch.max(F.log_softmax(inference[0].cpu(), dim=1), 1)
        for j in range(predicted.shape[0]):
            c_j = coords[j][0]
            for c in range(n_classes):
                segmentations[subj_id][c, c_j[0, 0]:c_j[0, 1],
                                       c_j[1, 0]:c_j[1, 1],
                                       c_j[2, 0]:c_j[2, 1]] += (predicted[j] == c)

    for i in segmentations.keys():
        segmentations[i] = torch.max(segmentations[i], 0)[1]
    return segmentations


n_classes = 60
infer_loader = infer_loaders['infer']

segmentations = voxel_majority_predict_from_subvolumes(
    infer_loader, [256, 256, 256], n_classes)
