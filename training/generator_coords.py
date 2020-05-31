import numpy as np
from scipy.stats import truncnorm


class CoordsGenerator:
    """
    Args:
        Generation random coordinates

    """

    def __init__(
        self, list_shape=None, list_sub_shape=None, mus=None, sigmas=None
    ):
        """
        Args:
            list_shape
            list_sub_shape
            mus
            sigmas
        """
        self.volume_shape = np.array(list_shape)
        self.subvolume_shape = np.array(list_sub_shape)

        self.half_subvolume_shape = self.subvolume_shape // 2
        if mus is None:
            mus = np.array(
                [
                    self.volume_shape[0] // 2,
                    self.volume_shape[0] // 2,
                    self.volume_shape[0] // 2,
                ]
            )
        if sigmas is None:
            sigmas = np.array(
                [
                    self.volume_shape[0] // 4,
                    self.volume_shape[0] // 4,
                    self.volume_shape[0] // 4,
                ]
            )
        self.truncnorm_coordinates = truncnorm(
            (self.half_subvolume_shape - mus + 1) / sigmas,
            (self.volume_shape - self.half_subvolume_shape - mus) / sigmas,
            loc=mus,
            scale=sigmas,
        )

    def _generator(self):
        xyz = np.round(self.truncnorm_coordinates.rvs(size=(1, 3))[0]).astype(
            "int"
        )
        xyz_start = xyz - self.half_subvolume_shape
        xyz_end = xyz + self.half_subvolume_shape
        xyz_coords = np.vstack((xyz_start, xyz_end)).T
        return xyz_coords

    def __generate_centered_nonoverlap_1d_grid(self):
        """
        Generates a centered nonoverlap grid.
        Grid will not cover the whole volume if the multiplier
        of the volume shape is not equal to subvolume shape.

        Args:
            length (int): volume side length
            step (int): subvolume side length
        """
        step = self.subvolume_shape[0]
        length = self.volume_shape[0]
        return [
            (c, c + step)
            for c in range((length % step) // 2, length - step + 1, step)
        ]

    def _generate_centered_nonoverlap_1d_grid(self):
        z = self.__generate_centered_nonoverlap_1d_grid()
        y = self.__generate_centered_nonoverlap_1d_grid()
        x = self.__generate_centered_nonoverlap_1d_grid()
        return np.array([[i, j, l] for i in z for j in y for l in x])

    def get_coordinates(self, mode="train", n_samples=100):
        """
        Args:
            n_samples: numbers of subsamples
            mode: mode ot training
        """
        if mode == "train":
            coord = [self._generator() for _ in range(n_samples)]
        else:
            coord = self._generate_centered_nonoverlap_1d_grid()
        return coord
