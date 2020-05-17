import collections

from reader import NiftiReader_Image, NiftiReader_Mask

import torch
from torch import nn
from torchvision import transforms

from catalyst.contrib.utils.pandas import read_csv_data
from catalyst.data import Augmentor, ListDataset, ReaderCompose
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    """
    Args:
        stage (str)
        mode (str)
    """

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
        """
        Args:
            stage (str)
            mode (str)
        """
        if mode == "train":
            Augmentor1 = Augmentor(
                dict_key="images",
                augment_fn=lambda x: torch.from_numpy(x).float(),
            )
            Augmentor2 = Augmentor(
                dict_key="labels", augment_fn=lambda x: torch.from_numpy(x)
            )
            return transforms.Compose([Augmentor1, Augmentor2])
        elif mode == "valid":
            Augmentor1 = Augmentor(
                dict_key="images",
                augment_fn=lambda x: torch.from_numpy(x).float(),
            )
            Augmentor2 = Augmentor(
                dict_key="labels", augment_fn=lambda x: torch.from_numpy(x)
            )
            return transforms.Compose([Augmentor1, Augmentor2])

    def get_datasets(
        self,
        stage: str,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None,
    ):
        """
        Args:
            stage (str)
            in_csv_train (str)
            in_csv_valid (str)
            in_csv_infer (str)
        """
        df, df_train, df_valid, df_infer = read_csv_data(
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
        )

        datasets = collections.OrderedDict()
        open_fn = ReaderCompose(
            readers=[
                NiftiReader_Image(input_key="images", output_key="images"),
                NiftiReader_Mask(input_key="labels", output_key="labels"),
            ]
        )

        for mode, source in zip(("train", "valid"), (df_train, df_valid)):
            if source is not None and len(source) > 0:
                datasets[mode] = ListDataset(
                    list_data=source,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(stage=stage, mode=mode),
                )

        return datasets
