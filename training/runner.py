from typing import List
from torchvision import transforms

from brain_dataset import BrainDataset
from reader import NiftiReader_Image, NiftiReader_Mask

import pandas as pd
import torch
from torch.utils.data import RandomSampler

from catalyst.dl import IRunner, SupervisedConfigRunner
from catalyst.contrib.utils.pandas import dataframe_to_list

from catalyst.data import Augmentor, ReaderCompose


class IRunnerMixin(IRunner):
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
        Augmentor1 = Augmentor(
            dict_key="images",
            augment_fn=lambda x: torch.from_numpy(x).float(),
        )
        Augmentor2 = Augmentor(
            dict_key="targets", augment_fn=lambda x: torch.from_numpy(x)
        )
        return transforms.Compose([Augmentor1, Augmentor2])

    def get_datasets(
        self,
        subvolume_shape: List[int],
        volume_shape: List[int],
        stage: str,
        in_csv_train: str = None,
        in_csv_valid: str = None,
        n_subvolumes: int = 128,
        **kwargs
    ):
        """
        Args:
            subvolume_shape: dimention of subvolume
            volume_shape: dimention of volume
            stage (str)
            in_csv_train (str): csv with training information
            in_csv_valid (str): csv with validation information
            in_csv_infer (str): csv with inference information
        """

        datasets = {}
        open_fn = ReaderCompose(
            [NiftiReader_Image(input_key="images", output_key="images"),
             NiftiReader_Mask(input_key="nii_labels", output_key="targets")]
        )

        for mode, source in zip(("train", "validation"), (in_csv_train,
                                                          in_csv_valid)):
            if source is not None and len(source) > 0:
                source_df = pd.read_csv(source)
                dataset = BrainDataset(
                    shared_dict={},
                    list_data=dataframe_to_list(source_df),
                    list_shape=volume_shape,
                    list_sub_shape=subvolume_shape,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(stage=stage, mode=mode),
                    n_subvolumes=n_subvolumes,
                    mode=mode,
                    input_key="images",
                    output_key="targets",
                )
                if mode in ["train", "validation"]:
                    sampler = RandomSampler(
                        data_source=dataset, replacement=True,
                        num_samples=len(source_df) * n_subvolumes
                    )

                datasets[mode] = {"dataset": dataset, "sampler": sampler}

        return datasets


class CustomSupervisedConfigRunner(IRunnerMixin, SupervisedConfigRunner):
    pass
