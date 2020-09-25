import nibabel as nib
import numpy as np
import joblib
import time

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
        #print("open_nifti_files time: ",  end-start)
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
        with open("./presets/label_protocol_unique.txt", "r") as f:
            t = f.read()

        labels = [int(x) for x in t.split(",")]
        img = nib.load(image_name, mmap=False)
        img = img.get_fdata(dtype=np.float32)
        new_img = np.zeros([256, 256, 256])
        new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        segmentation = np.zeros([256, 256, 256])

        for i, l in enumerate(labels):
            mask = np.equal(l, new_img)
            segmentation[mask] = i

        end = time.time()
        #print("label_files time: ",  end-start)
        output = {self.output_key: segmentation}
        return output


class JoblibReader(ReaderSpec):
    """
    Joblib reader abstraction for saved numpy one hot labels
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
        data = joblib.load(image_name)
        end = time.time()
        #print(image_name)
        #print("open_joblib_files time: ",  end-start)

        output = {self.output_key: data}
        return output
