from typing import List

from brain_dataset import BrainDataset
from Collate_generator import CollateGeneratorFn
from reader import NiftiReader_Image, NiftiReader_Mask

import torch
from torchvision import transforms

from catalyst.contrib.utils.pandas import read_csv_data
from catalyst.data import Augmentor, ReaderCompose
from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    """
    Args:
        stage (str)
        mode (str)
    """

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
        subvolume_shape: List[int],
        volume_shape: List[int],
        stage: str,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        in_csv_infer: str = None,
        n_samples: int = 100,
    ):
        """
        Args:
            subvolume_shape: dimention of subvolume
            volume_shape: dimention of volume
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

        datasets = {}
        open_fn = ReaderCompose(
            readers=[
                NiftiReader_Image(input_key="images", output_key="images"),
                NiftiReader_Mask(input_key="labels", output_key="labels"),
            ]
        )

        for mode, source in zip(("train", "valid"), (df_train, df_valid)):
            if source is not None and len(source) > 0:

                datasets[mode] = {
                    "dataset": BrainDataset(
                        list_data=source,
                        list_shape=subvolume_shape,
                        list_sub_shape=volume_shape,
                        open_fn=open_fn,
                        dict_transform=self.get_transforms(
                            stage=stage, mode=mode
                        ),
                        mode=mode,
                        n_samples=n_samples,
                        input_key="images",
                        output_key="labels",
                    ),
                    "collate_fn": CollateGeneratorFn("images", "labels"),
                }

        return datasets
