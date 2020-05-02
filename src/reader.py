from catalyst.data import ReaderSpec
import nibabel as nib
import os
import numpy as np
import math
import ast

import numpy as np

class NiftiReader_Image(ReaderSpec):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from a `csv` dataset.
    """
    def __init__(
        self,
        input_key: str,
        coords: list,
        output_key: str,
        rootpath: str = None
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            grayscale (bool): flag if you need to work only
                with grayscale images
        """
        super().__init__(input_key, output_key)
        self.rootpath = rootpath
        self.coords = coords

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        coords = ast.literal_eval(element[self.coords])
        image_name = str(element[self.input_key])
        img = np.load(image_name)
        subvolume_shape = np.array([64,64,64])
        x = np.zeros((1,subvolume_shape[0],subvolume_shape[1],subvolume_shape[2]))
        x[0,:subvolume_shape[0],:subvolume_shape[1],:subvolume_shape[2]] = img[
                    coords[0][0]:coords[0][1],
                    coords[1][0]:coords[1][1],
                    coords[2][0]:coords[2][1]]

        output = {self.output_key: x.astype(np.float32)}
        return output

class NiftiReader_Mask(ReaderSpec):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from a `csv` dataset.
    """
    def __init__(
        self,
        input_key: str,
        coords,
        output_key: str,
        rootpath: str = None
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            grayscale (bool): flag if you need to work only
                with grayscale images
        """
        super().__init__(input_key, output_key)
        self.rootpath = rootpath
        self.coords = coords

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        coords = ast.literal_eval(element[self.coords])
        subvolume_shape = np.array([64,64,64])
        image_name = str(element[self.input_key])
        label = np.load(image_name, allow_pickle=True)
        y = np.zeros([label.shape[0],subvolume_shape[0], subvolume_shape[1], subvolume_shape[2]])
        y[:,:subvolume_shape[0],:subvolume_shape[1],:subvolume_shape[2]] = label[:,
                    coords[0][0]:coords[0][1],
                    coords[1][0]:coords[1][1],
                    coords[2][0]:coords[2][1]]
        output = {self.output_key: y}
        return output