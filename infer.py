import nibabel as nib
import numpy as np
from training.generator_coords import CoordsGenerator

import torch


class Predictor:
    """Only useful for serving... other methods would use a dataloader"""

    def __init__(
        self, model, volume_shape, subvolume_shape, n_subvolumes, n_classes
    ):
        """Docs."""
        self.model = model
        self.volume_shape = volume_shape
        self.subvolume_shape = subvolume_shape
        self.n_subvolumes = n_subvolumes
        self.n_classes = n_classes

        self.generator = CoordsGenerator(
            self.volume_shape, self.subvolume_shape
        )
        if len(self.generator.get_coordinates(mode="test")) > n_subvolumes:
            raise ValueError(
                "n_subvolumes must be at least {coords_len}".format(
                    coords_len=len(self.generator.get_coordinates(mode="test"))
                )
            )
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

    def generate_coords(self):
        """Generate coordinate for prediction"""
        coords_list = []
        for i in range(self.n_subvolumes):
            coords = self.generator.get_coordinates(mode="test")
            if i >= len(coords):
                coords = self.generator.get_coordinates()
                coords_list.append(coords)
            else:
                coords_list.append(np.expand_dims(coords[i], 0))

        return coords_list

    def preprocess_image(self, img):
        """Unit interval preprocessing"""
        img = (img - img.min()) / (img.max() - img.min())
        new_img = np.zeros(self.volume_shape)
        new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        return new_img

    def predict(self, image_path):
        """Predict segmentation given an image_path"""
        img = nib.load(image_path)
        img = img.get_fdata()
        normalized_img = self.preprocess_image(img)
        coords_list = self.generate_coords()
        one_hot_predicted_segmentation = torch.zeros(
            tuple(np.insert(self.volume_shape, 0, self.n_classes)),
            dtype=torch.uint8,
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for coords in coords_list:
                input_slice = np.expand_dims(
                    normalized_img[
                        coords[0][0][0] : coords[0][0][1],
                        coords[0][1][0] : coords[0][1][1],
                        coords[0][2][0] : coords[0][2][1],
                    ],
                    0,
                )
                torch_slice = torch.from_numpy(
                    np.expand_dims(input_slice, 0).astype(np.float32)
                ).to(self.device)
                _, predicted = torch.max(
                    torch.nn.functional.log_softmax(
                        self.model(torch_slice), dim=1
                    ),
                    1,
                )

                for j in range(predicted.shape[0]):
                    c_j = coords[j]
                    for c in range(self.n_classes):
                        one_hot_predicted_segmentation[
                            c,
                            c_j[0, 0] : c_j[0, 1],
                            c_j[1, 0] : c_j[1, 1],
                            c_j[2, 0] : c_j[2, 1],
                        ] += (predicted[j] == c)

            predicted_segmentation = torch.max(
                one_hot_predicted_segmentation, 0
            )[1]
        return predicted_segmentation
