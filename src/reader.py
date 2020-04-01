from catalyst.data import ReaderSpec
import nibabel as nib
import os
import numpy as np
import math

def zero_pad_array(array, d, w, h):
    '''Function to pad tensors with zeros to same batch_size
    Args:
        array (numpy array): The tensor that is to be padded
        d (int): Size of this function output in first dimension
        w (int): Size of this function output in second dimension
        h (int): Size of this function output in third dimension
    '''
    x, y, z = array.shape

    pad_x1= 0
    pad_x2= 0
    pad_y1= 0
    pad_y2= 0
    pad_z1= 0
    pad_z2= 0

    d_x = d-x
    d_y = w-y
    d_z = h-z

    if d_x%2 ==0:
        pad_x1 = pad_x2 = d_x//2
    else:
        pad_x1 = math.floor(d_x/2)
        pad_x2 = math.ceil(d_x/2)

    if d_y%2 ==0:
        pad_y1 = pad_y2 = d_y//2
    else:
        pad_y1 = math.floor(d_y/2)
        pad_y2 = math.ceil(d_y/2)

    if d_z%2 ==0:
        pad_z1 = pad_z2 = d_z//2
    else:
        pad_z1 = math.floor(d_z/2)
        pad_z2 = math.ceil(d_z/2)

    pad_width = ((pad_x1,pad_x2),(pad_y1,pad_y2),(pad_z1,pad_z2))
    ret = np.pad(array,pad_width, mode='constant', constant_values=0)
    ret = np.array([list(ret)])
    return ret



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
        print(element)
        image_name = str(element[self.input_key])

        # img = nib.load(
        #     os.path.join(self.rootpath, image_name)
        # )
        img = nib.load(image_name)
        img = img.get_fdata(caching='unchanged')
        img = zero_pad_array(img, 256, 256, 256)

        output = {self.output_key: img}
        return output