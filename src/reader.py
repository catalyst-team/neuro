from catalyst.data import ReaderSpec
import numpy as np


class NiftiReader(ReaderSpec):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from a `csv` dataset.
    """
    def __init__(
        self,
        input_key: str,
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

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        image = np.frombuffer(element[self.input_key], dtype=np.uint8)
        output = {self.output_key: image}
        return output


