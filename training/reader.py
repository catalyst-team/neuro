import time

import joblib
import nibabel as nib
import numpy as np

from catalyst.data import ReaderSpec


class NiftiReader_Image(ReaderSpec):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images
    from a `csv` dataset.
    """

    def __init__(
        self, input_key: str, output_key: str, rootpath: str = None,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            coords (list): crop coordinaties
        """
        super().__init__(input_key, output_key)
        self.rootpath = rootpath

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        start = time.time()
        image_name = str(element[self.input_key])
        img = nib.load(image_name)
        img = img.get_fdata(dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        new_img = np.zeros([256, 256, 256])
        new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img

        output = {self.output_key: new_img.astype(np.float32)}
        end = time.time()
        return output


class NiftiReader_Mask(ReaderSpec):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from
    a `csv` dataset.
    """

    def __init__(self, input_key: str, output_key: str, rootpath: str = None):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            coords (list): crop coordinaties
        """
        super().__init__(input_key, output_key)
        self.rootpath = rootpath

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        start = time.time()
        image_name = str(element[self.input_key])
        img = nib.load(image_name, mmap=False)
        img = img.get_fdata(dtype=np.float32)
        output = {self.output_key: img}
        return output
