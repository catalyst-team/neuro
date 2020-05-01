import collections
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from scipy.stats import truncnorm
from catalyst.contrib.utils.pandas import read_csv_data
from reader import NiftiReader_Mask, NiftiReader_Image
from catalyst.data import (
    ListDataset,
    ReaderCompose,
)
from catalyst.data import Augmentor
from torchvision import transforms
from catalyst.dl import ConfigExperiment


volume_shape = np.array([256,256,256])
subvolume_shape = np.array([64,64,64])

half_subvolume_shape = subvolume_shape // 2

mus = np.array(
    [volume_shape[0] // 2,
    volume_shape[0] // 2,
    volume_shape[0] // 2]
)
sigmas = np.array(
    [volume_shape[0] // 4,
    volume_shape[0] // 4,
    volume_shape[0] // 4]
)

truncnorm_coordinates = truncnorm(
    (half_subvolume_shape - mus + 1) / sigmas,
    (volume_shape - half_subvolume_shape - mus) / sigmas,
    loc=mus, scale=sigmas
)

def coords_generator():
    xyz = np.round(truncnorm_coordinates.rvs(size=(1, 3))[0]).astype('int')
    xyz_start = xyz - half_subvolume_shape
    xyz_end = xyz + half_subvolume_shape
    xyz_coords = np.vstack((xyz_start, xyz_end)).T
    return xyz_coords


def generation_coordinates(image, n_samples):
    for coords in [coords_generator() for k in range(n_samples)]:
        img = np.zeros(len(image.shape[0]), subvolume_shape[0], subvolume_shape[1], subvolume_shape[2])
        for k in range(len(image.shape[0])):
            img[k:,:subvolume_shape[0],
            :subvolume_shape[1],:subvolume_shape[2]] = image[coords[0][0]: coords[0][1],
            coords[1][0]: coords[1][1],
            coords[2][0]: coords[2][1]]
    return pd.DataFrame(out_data)

class Experiment(ConfigExperiment):



    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        # if stage in ["debug", "stage1"]:
        #     for param in model_.encoder.parameters():
        #         param.requires_grad = False
        # elif stage == "stage2":
        #     for param in model_.encoder.parameters():
        #         param.requires_grad = True
        return model_
    def get_transforms(self, stage: str = None, mode: str = None):
        if mode == "train":
            Augmentor1 = Augmentor(dict_key = "images" ,augment_fn=lambda x: torch.from_numpy(x).float())
            Augmentor2 = Augmentor(dict_key = "labels" ,augment_fn=lambda x: torch.from_numpy(x))
            return transforms.Compose([Augmentor1, Augmentor2])
        elif mode == "valid":
            Augmentor1 = Augmentor(dict_key = "images" ,augment_fn=lambda x: torch.from_numpy(x).float())
            Augmentor2 = Augmentor(dict_key = "labels" ,augment_fn=lambda x: torch.from_numpy(x))
            return transforms.Compose([Augmentor1, Augmentor2])
        

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None
    ):
        df, df_train, df_valid, df_infer = read_csv_data(in_csv_train=in_csv_train, in_csv_valid=in_csv_valid, in_csv_infer=in_csv_infer)                                                                   
        datasets = collections.OrderedDict()
        open_fn = ReaderCompose(
            readers=[
                NiftiReader(
                    input_key="images", output_key="images"),
                NiftiReader(
                    input_key="labels", output_key="labels"),
            ]
        )

        for mode, source in zip(
            ("train", "valid"), (df_train, df_valid)
        ):
            if source is not None and len(source) > 0:
                datasets[mode] = ListDataset(
                    list_data=source,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(stage=stage, mode=mode))

        return datasets