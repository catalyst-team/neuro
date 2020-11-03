from typing import Any, Callable, List, Union
from pathlib import Path
import time

from generator_coords import CoordsGenerator
import numpy as np
from multiprocessing import Manager

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
        dict_transform: Callable = None,
        n_samples: int = 100,
        mode: str = "train",
        input_key: str = "images",
        output_key: str = "targets",
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
        self.n_samples = n_samples
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

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        if self.mode != 'train':
            # 216 is a hardcode for all the different nonoverlapping coords for
            # a 38x38x38 cross_section from 256x256x256
            return len(self.data) * 216
        else:
            return len(self.data)

    def __getitem__(self, index: int) -> Any:
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            List of elements by index
        """

        start = time.time()
        coords = self.generator.get_coordinates(mode=self.mode)
        end = time.time()

        if self.mode != 'train':
            coords = np.expand_dims(coords[index//216], 0)
            item = self.data[index // 216]
        else:
            item = self.data[index]
        start = time.time()
        dict_ = self.open_fn(item)
        end = time.time()

        start = time.time()
        sample_dict = self.__crop__(dict_, coords)
        end = time.time()

        return sample_dict

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
        output = self.shared_dict
        output_labels_list = []
        output_images_list = []
        for start_end in coords:
            for key, _ in dict_.items():
                if key == self.input_key:
                    output_images_list.append(np.expand_dims(dict_[key][
                        start_end[0][0] : start_end[0][1],
                        start_end[1][0] : start_end[1][1],
                        start_end[2][0] : start_end[2][1],
                    ], 0))

                elif key == self.output_key:
                    output_labels_list.append(np.expand_dims(dict_[key][
                        start_end[0][0] : start_end[0][1],
                        start_end[1][0] : start_end[1][1],
                        start_end[2][0] : start_end[2][1],
                    ], 0))

        output_images = np.concatenate(output_images_list)
        output_labels = np.concatenate(output_labels_list)
        output[self.input_key] = output_images
        output[self.output_key] = output_labels.squeeze().astype(np.int64)
        return self.dict_transform(output)
