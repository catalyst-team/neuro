import ast

import numpy as np
import nibabel as nib

from catalyst.data import ReaderSpec


class NiftiReader_Image(ReaderSpec):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images
    from a `csv` dataset.
    """

    def __init__(
        self,
        input_key: str,
        coords: list,
        output_key: str,
        rootpath: str = None,
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
        img = nib.load(image_name)
        img = img.get_fdata(dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 255.0
        new_img = np.zeros([256, 256, 256])
        new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        subvolume_shape = np.array([64, 64, 64])
        x = np.zeros(
            (1, subvolume_shape[0], subvolume_shape[1], subvolume_shape[2])
        )
        x[
            0, : subvolume_shape[0], : subvolume_shape[1], : subvolume_shape[2]
        ] = new_img[
            coords[0][0] : coords[0][1],
            coords[1][0] : coords[1][1],
            coords[2][0] : coords[2][1],
        ]

        output = {self.output_key: x.astype(np.float32)}
        return output


class NiftiReader_Mask(ReaderSpec):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from
    a `csv` dataset.
    """

    def __init__(
        self, input_key: str, coords, output_key: str, rootpath: str = None
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
        subvolume_shape = np.array([64, 64, 64])
        image_name = str(element[self.input_key])
        with open("/home/Bekovmi/neuro/presets/label_protocol_unique.txt", "r") as f:
            t = f.read()

        labels = [int(x) for x in t.split(",")]
        img = nib.load(image_name, mmap=False)
        img = img.get_fdata(dtype=np.float32)
        segmentation = np.zeros([len(labels), 256, 256, 256])
        k = 0
        for l in labels:
            segmentation[k, : img.shape[0], : img.shape[1], : img.shape[2]] = (
                    img == l
            )
            k += 1
        data = segmentation.astype("uint8")
        y = np.zeros(
            [
                data.shape[0],
                subvolume_shape[0],
                subvolume_shape[1],
                subvolume_shape[2],
            ]
        )
        y[
            :, : subvolume_shape[0], : subvolume_shape[1], : subvolume_shape[2]
        ] = data[
            :,
            coords[0][0] : coords[0][1],
            coords[1][0] : coords[1][1],
            coords[2][0] : coords[2][1],
        ]
        output = {self.output_key: y}
        return output
