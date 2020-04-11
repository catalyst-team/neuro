from catalyst.data import ReaderSpec
import nibabel as nib
import os
import numpy as np
import math

class NiftiReader_Image(ReaderSpec):
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
        image_name = str(element[self.input_key])

        # img = nib.load(
        #     os.path.join(self.rootpath, image_name)
        # )
        y = np.zeros((1,256,256,256))
        img = nib.load(image_name, mmap=False)
        img = img.get_fdata(dtype=np.float32)
        img = (img - img.min())/(img.max()-img.min())
        img = img*255.0
        img = np.transpose(img,(2, 0, 1))
        
        y[0,:img.shape[0],:img.shape[1],:img.shape[2]] = img
        output = {self.output_key: y}
        return output

class NiftiReader_Mask(ReaderSpec):
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
        image_name = str(element[self.input_key])

        # img = nib.load(
        #     os.path.join(self.rootpath, image_name)
        # )
        
        img = nib.load(image_name)

        img = nib.load(image_name, mmap=False)
        img = img.get_fdata(dtype=np.float32)
        img = np.transpose(img,(2, 0, 1))
        with open("./src/label_protocol.txt","r") as f:
            t = f.read()
        labels = [int(x) for x in t.split(",")]
        segmentation = np.zeros((len(labels),256,256,256))
        segmentation[0,:,:,:] = np.ones([256,256,256])
        for k in range(1,len(labels)):
            seg_one = img == labels[k]
            segmentation[k,:img.shape[0],:img.shape[1],:img.shape[2]] = seg_one
            segmentation[0,:,:,:] = segmentation[0,:,:,:] - segmentation[k,:,:,:]
        output = {self.output_key: segmentation.astype('int64')}
        return output



