from typing import List

from brain_dataset import BrainDataset
from reader import NiftiReader_Image, NiftiReader_Mask

import torch
from torchvision import transforms

from catalyst.contrib.utils.pandas import read_csv_data
from catalyst.data import Augmentor, ReaderCompose
from catalyst.dl import ConfigExperiment
from torch.utils.data import RandomSampler
from multiprocessing import Manager


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
                dict_key="targets", augment_fn=lambda x: torch.from_numpy(x)
            )
            return transforms.Compose([Augmentor1, Augmentor2])
        elif mode == "valid":
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
        n_samples: int = 100,
        max_batch_size: int = 3,
        **kwargs
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


        for mode, source in zip(("train", "valid"), (df_train, df_valid)):
            if source is not None and len(source) > 0:
                dataset = BrainDataset(
                    shared_dict=manager.dict(),
                    list_data=source, list_shape=volume_shape, list_sub_shape=subvolume_shape,
                    open_fn=open_fn, dict_transform=self.get_transforms(stage=stage, mode=mode),
                    mode=mode, n_samples=n_samples, input_key="images",
                    output_key="targets")
                train_random_sampler = RandomSampler(data_source=dataset,
                                                     replacement=True,
                                                     num_samples=80 * 128)
                val_random_sampler = RandomSampler(data_source=dataset,
                                                   replacement=True,
                                                   num_samples=20 * 216)
                if mode == 'train':
                    datasets[mode] = {"dataset": dataset,
                                      "sampler": train_random_sampler}
                else:
                    datasets[mode] = {"dataset": dataset,
                                      "sampler": val_random_sampler}
        return datasets
