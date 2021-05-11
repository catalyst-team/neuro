import nibabel as nib
from typing import Optional, List
import numpy as np

from catalyst.contrib.data.reader import IReader


class NiftiReader(IReader):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from
    a `csv` dataset.
    """

    def __init__(
        self, input_key: str, output_key: Optional[str] = None, rootpath: Optional[str] = None
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        image_name = str(element[self.input_key])
        img = nib.load(image_name)
        output = {self.output_key: img}
        return output


class NiftiFixedVolumeReader(NiftiReader):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images
    from a `csv` dataset.
    """

    def __init__(
        self, input_key: str, output_key: str, rootpath: str = None,
        volume_shape: List = None
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            coords (list): crop coordinaties
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        if volume_shape is None:
            volume_shape = [256, 256, 256]
        self.volume_shape = volume_shape

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        image_name = str(element[self.input_key])
        img = nib.load(image_name)
        img = img.get_fdata()
        img = (img - img.min()) / (img.max() - img.min())
        new_img = np.zeros(self.volume_shape)
        new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        output = {self.output_key: new_img}
        return output
