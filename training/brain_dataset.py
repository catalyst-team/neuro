from typing import Any, Callable, List, Union
from multiprocessing import Manager
from pathlib import Path

from generator_coords import CoordsGenerator
import numpy as np

from torch.utils.data import Dataset

_Path = Union[str, Path]


class BrainDataset(Dataset):
    """General purpose dataset class with several data sources `list_data`."""

    def __init__(
        self,
        shared_dict,
        list_data: List[int],
        list_shape: List[int],
        list_sub_shape: List[int],
        open_fn: Callable,
        n_subvolumes: int = None,
        dict_transform: Callable = None,
        mode: str = "train",
        input_key: str = "images",
        output_key: str = "targets",
    ):
        """
        Args:
            list_data (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            list_shape (List[int]):
            list_sub_shape (List[int]):
            open_fn (callable): function, that can open your
                annotations dict and
                transfer it to data, needed by your network
                (for example open image by path, or tokenize read string.)
            dict_transform (callable): transforms to use on dict.
                (for example normalize image, add blur, crop/resize/etc)
        """
        self.shared_dict = shared_dict
        self.data = list_data
        self.open_fn = open_fn
        self.generator = CoordsGenerator(
            list_shape=list_shape, list_sub_shape=list_sub_shape
        )
        self.mode = mode
        self.dict_transform = (
            dict_transform if dict_transform is not None else lambda x: x
        )
        self.input_key = input_key
        self.output_key = output_key
        self.subvolume_shape = np.array(list_sub_shape)
        self.subjects = len(self.data)
        self.n_subvolumes = n_subvolumes

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        if self.mode not in ["train", "validation"]:
            return len(self.n_subvolumes) * len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, index: int) -> Any:
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            List of elements by index
        """

        if self.mode not in ["train", "validation"]:
            coords = self.generator.get_coordinates(mode='test')
            if index > len(coords):
                coords = self.generator.get_coordinates()

            item = self.data[index // len(self.n_subvolumes)]

        else:
            item = self.data[index]
            coords = self.generator.get_coordinates()
        dict_ = self.open_fn(item)
        sample_dict = self.__crop__(dict_, coords)

        return sample_dict

    def __crop__(self, dict_, coords):
        """Get crop of images.
        Args:
            dict_ (List[Dict]): list of dicts, that stores
                you data annotations,
                (for example path to images, labels, bboxes, etc.)
            coords (callable): coords of crops

        Returns:
            crop images
        """
        output = self.shared_dict
        output_labels_list = []
        output_images_list = []
        for start_end in coords:
            for key, dict_key in dict_.items():
                if key == self.input_key:
                    output_images_list.append(
                        np.expand_dims(
                            dict_key[
                                start_end[0][0] : start_end[0][1],
                                start_end[1][0] : start_end[1][1],
                                start_end[2][0] : start_end[2][1],
                            ],
                            0,
                        )
                    )

                elif key == self.output_key:
                    output_labels_list.append(np.expand_dims(
                        dict_key[start_end[0][0] : start_end[0][1],
                                 start_end[1][0] : start_end[1][1],
                                 start_end[2][0] : start_end[2][1]],
                        0,))

        output_images = np.concatenate(output_images_list)
        output_labels = np.concatenate(output_labels_list)
        output[self.input_key] = output_images
        output[self.output_key] = output_labels.squeeze().astype(np.int64)
        output['coords'] = coords
        return self.dict_transform(output)
