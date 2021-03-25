from typing import List
from multiprocessing import Manager

from brain_dataset import BrainDataset
from reader import NiftiReader_Image, NiftiReader_Mask

import torch
from torch.utils.data import RandomSampler, SequentialSampler
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
        in_csv_infer: str = None,
        n_samples: int = 128,
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
        manager = Manager()
        df, df_train, df_valid, df_infer = read_csv_data(
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
        )

        datasets = {}
        open_fn = ReaderCompose(
            readers=[
                NiftiReader_Image(input_key="images", output_key="images"),
                NiftiReader_Mask(input_key="nii_labels", output_key="targets"),
            ]
        )

        sampler_dict = {}
        for mode, source in zip(("train", "valid", "infer"),
                                (df_train, df_valid, df_infer)):
            if source is not None and len(source) > 0:
                dataset = BrainDataset(
                    shared_dict=manager.dict(),
                    list_data=source,
                    list_shape=volume_shape,
                    list_sub_shape=subvolume_shape,
                    open_fn=open_fn,
                    dict_transform=self.get_transforms(stage=stage, mode=mode),
                    mode=mode,
                    n_samples=n_samples,
                    input_key="images",
                    output_key="targets",
                )
                if mode in ["train", "valid"]:
                    sampler = RandomSampler(
                        data_source=dataset, replacement=True,
                        num_samples=len(source) * n_samples

                    )
                else:
                    sampler = SequentialSampler(data_source=dataset)

                datasets[mode] = {"dataset": dataset, "sampler": sampler}

        return datasets
