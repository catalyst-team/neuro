from typing import Any, Callable, List, Union
from pathlib import Path

from generator_coords import CoordsGenerator
import numpy as np

from torch.utils.data import Dataset

_Path = Union[str, Path]


class BrainDataset(Dataset):
    """General purpose dataset class with several data sources `list_data`."""

    def __init__(
        self,
        list_data: List[int],
        list_shape: List[int],
        list_sub_shape: List[int],
        open_fn: Callable,
        dict_transform: Callable = None,
        mode: str = "train",
        n_samples: int = 100,
        input_key: str = "images",
        output_key: str = "labels",
    ):
        """
        Args:
            list_data (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            open_fn (callable): function, that can open your
                annotations dict and
                transfer it to data, needed by your network
                (for example open image by path, or tokenize read string.)
            dict_transform (callable): transforms to use on dict.
                (for example normalize image, add blur, crop/resize/etc)
        """
        self.data = list_data
        self.open_fn = open_fn
        self.generator = CoordsGenerator(
            list_shape=list_shape, list_sub_shape=list_sub_shape
        )
        self.mode = mode
        self.n_samples = n_samples
        self.dict_transform = (
            dict_transform if dict_transform is not None else lambda x: x
        )
        self.input_key = input_key
        self.output_key = output_key
        self.subvolume_shape = np.array(list_sub_shape)

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            List of elements by index
        """
        item = self.data[index]
        dict_ = self.open_fn(item)
        coords = self.generator.get_coordinates(
            mode=self.mode, n_samples=self.n_samples
        )
        list_image = [self.__crop__(dict_, coord) for coord in coords]
        return list_image

    def __crop__(self, dict_, coords):
        """Get crop of images.
        Args:
            dict_ (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            coords (callable): coords od crops
        Returns:
            crop images
        """
        output = {}
        for key, _ in dict_.items():
            if key == self.input_key:

                x = np.zeros(
                    [
                        1,
                        self.subvolume_shape[0],
                        self.subvolume_shape[1],
                        self.subvolume_shape[2],
                    ]
                )
                x[
                    0,
                    : self.subvolume_shape[0],
                    : self.subvolume_shape[1],
                    : self.subvolume_shape[2],
                ] = dict_[key][
                    coords[0][0] : coords[0][1],
                    coords[1][0] : coords[1][1],
                    coords[2][0] : coords[2][1],
                ]
                output[key] = x
            elif key == self.output_key:
                y = np.zeros(
                    [
                        1,
                        106,
                        self.subvolume_shape[0],
                        self.subvolume_shape[1],
                        self.subvolume_shape[2],
                    ]
                )
                y[
                    0,
                    :,
                    : self.subvolume_shape[0],
                    : self.subvolume_shape[1],
                    : self.subvolume_shape[2],
                ] = dict_[key][
                    :,
                    coords[0][0] : coords[0][1],
                    coords[1][0] : coords[1][1],
                    coords[2][0] : coords[2][1],
                ]
                output[key] = y
        return self.dict_transform(output)
