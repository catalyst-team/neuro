from urllib.request import urlopen

import albumentations as A
import cv2
import numpy as np
import torch

from ..dataset import CLASSES
from ../model import MNISTNet


class Predictor():

    def __init__(self, checkpoint, use_gpu=False):
        assert not use_gpu, "We're not using gpu predictor in this tutorial"

        self.model = MNISTNet(num_classes=len(CLASSES))
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.model.eval()

    @staticmethod
    def _prepare_img(url):
        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        img = cv2.resize(img, (28, 28)) - 255
        img = A.Normalize()(image=img)["image"]
        return img

    @staticmethod
    def _prepare_batch(img):
        img = np.moveaxis(img, -1, 0)
        vec = torch.from_numpy(img)
        batch = torch.unsqueeze(vec, 0)
        return batch

    def predict(self, url):
        img = self._prepare_img(url)
        batch = self._prepare_batch(img)
        out = self.model.forward(batch)
        out = out.detach().cpu().numpy()
        return CLASSES[np.argmax(out)]
