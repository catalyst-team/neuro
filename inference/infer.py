import numpy as np
import torch.nn as nn
import torch 
from catalyst.dl import utils
import nibabel as nib

def generate_centered_nonoverlap_1d_grid(length, step):
    """
    Generates a centered nonoverlap grid.
    Grid will not cover the whole volume if the multiplier 
    of the volume shape is not equal to subvolume shape.
    ARguments:
        length (int): volume side length
        step (int): subvolume side length
    """
    return [(c, c + step) for c in range(
        (length % step) // 2, length - step + 1, step)]

class Predictor:
    def __init__(self, model_path):
        self.m = nn.Softmax()
        self.model = utils.load_traced_model(model_path).cuda()
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        volume_shape = np.array([256,256,256])
        subvolume_shape = np.array([32,32,32])
        z = generate_centered_nonoverlap_1d_grid(volume_shape[0],subvolume_shape[0])
        y = generate_centered_nonoverlap_1d_grid(volume_shape[1],subvolume_shape[1])
        x = generate_centered_nonoverlap_1d_grid(volume_shape[2],subvolume_shape[2])
        self.coord = np.array([[i, j, l] for i in z for j in y for l in x])
    def predict(self,image_name):
        img = nib.load(image_name)
        img = img.get_fdata(dtype=np.float32)
        img = (img - img.min())/(img.max()-img.min())
        img = img*255.0
        new_img = np.zeros([1, 256,256,256])
        new_img[0, :img.shape[0], :img.shape[1], :img.shape[2]] = img
        prediction_mask = np.zeros([256,256,256])
        for i in range(len(self.coord)):
            z1, z2 = self.coord[i][0][0], self.coord[i][0][1]
            y1, y2 = self.coord[i][1][0], self.coord[i][1][1]
            x1, x2 = self.coord[i][2][0], self.coord[i][2][1]
            inputs = torch.from_numpy(new_img[0, z1:z2, y1:y2, x1:x2]).float()
            inputs = inputs.unsqueeze(0).to(self.device) 
            output = self.m(self.model(inputs)[0]).cpu().numpy()
            prediction_mask[z1:z2, y1:y2, x1:x2] = output
        return prediction_mask
