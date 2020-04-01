import collections
import json
import pandas as pd

import torch
from torch import nn

from catalyst.contrib.utils.pandas import read_csv_data
from .reader import NiftiReader
from catalyst.data import (
    ListDataset,
    ReaderCompose,
)

from catalyst.dl import ConfigExperiment


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
            return torch.from_numpy
        elif mode == "valid":
            return torch.from_numpy
        

    def get_datasets(
        self,
        stage: str,
        datapath: str = None,
        in_csv: str = None,
        in_csv_train: str = None,
        in_csv_valid: str = None,
    ):

        df_train = pd.read_csv(in_csv_train)
        df_valid = pd.read_csv(in_csv_valid)
        datasets = collections.OrderedDict()
        open_fn = ReaderCompose(
            readers=[
                NiftiReader(
                    input_key="images", output_key="images"
                ),
                NiftiReader(
                    input_key="labels", output_key="labels"
                ),
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