import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

volume_shape = np.array([256, 256, 256])
subvolume_shape = np.array([64, 64, 64])

half_subvolume_shape = subvolume_shape // 2

mus = np.array(
    [volume_shape[0] // 2, volume_shape[0] // 2, volume_shape[0] // 2]
)
sigmas = np.array(
    [volume_shape[0] // 4, volume_shape[0] // 4, volume_shape[0] // 4]
)

truncnorm_coordinates = truncnorm(
    (half_subvolume_shape - mus + 1) / sigmas,
    (volume_shape - half_subvolume_shape - mus) / sigmas,
    loc=mus,
    scale=sigmas,
)

with open("./presets/label_protocol_unique.txt", "r") as f:
    t = f.read()
LABELS = [int(x) for x in t.split(",")]


def coords_generator():
    """
    Args:
    """
    xyz = np.round(truncnorm_coordinates.rvs(size=(1, 3))[0]).astype("int")
    xyz_start = xyz - half_subvolume_shape
    xyz_end = xyz + half_subvolume_shape
    xyz_coords = np.vstack((xyz_start, xyz_end)).T
    return xyz_coords


def save_images(filename):
    """
    Args:
        filename (str): path to filename
    """
    img = nib.load(filename)
    path = os.path.dirname(filename)
    img = img.get_fdata(dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255.0
    new_img = np.zeros([256, 256, 256])
    new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
    np.save(os.path.join(path, "prepared_t1weighted.npy"), new_img)


def save_segmentation(filename):
    """
    Args:
        filename (str): path to filename
    """
    global LABELS

    path = os.path.dirname(filename)
    img = nib.load(filename, mmap=False)
    img = img.get_fdata(dtype=np.float32)
    segmentation = np.zeros([len(LABELS), 256, 256, 256])
    k = 0
    for l in LABELS:
        segmentation[k, : img.shape[0], : img.shape[1], : img.shape[2]] = (
            img == l
        )
        k += 1
    data = segmentation.astype("uint8")
    np.save(os.path.join(path, "prepared_labels.DKT31.manual.npy"), data)


def find_sample(path):
    """
    Args:
        path (str): path to mri images
    """
    labels_data = {"images": [], "labels": []}
    t = 0
    for case in os.listdir(path):
        if case.startswith("."):
            continue
        case_folder = os.path.join(path, case)
        for person in os.listdir(case_folder):
            if person.startswith("."):
                continue
            person_folder = os.path.join(case_folder, person)
            for sample in os.listdir(person_folder):
                if sample.startswith("."):
                    continue
                if sample == "t1weighted.nii":
                    save_images(os.path.join(person_folder, sample))
                    labels_data["images"].append(
                        os.path.join(person_folder, "t1weighted.npy")
                    )
                if sample == "labels.DKT31.manual+aseg.nii":
                    save_segmentation(os.path.join(person_folder, sample))
                    labels_data["labels"].append(
                        os.path.join(
                            person_folder, "prepared_labels.DKT31.manual.npy"
                        )
                    )
                    print(t)
                    t += 1

    return pd.DataFrame(labels_data)


def generation_coordinates(data, n_samples):
    """
    Args:
        datapath (dataframe): dataframe with columns "images","labels"
        n_samples (int): numbers of samples
    """
    out_data = {"images": [], "labels": [], "coords": [], "split": []}
    for i in range(len(data)):
        if i < len(data) * 0.8:
            for coords in [coords_generator() for k in range(n_samples)]:
                out_data["images"].append(data.iloc[i, 0])
                out_data["labels"].append(data.iloc[i, 1])
                out_data["coords"].append(coords.tolist())
                out_data["split"].append(0)
        else:
            coords = coords_generator()
            out_data["images"].append(data.iloc[i, 0])
            out_data["labels"].append(data.iloc[i, 1])
            out_data["coords"].append(coords.tolist())
            out_data["split"].append(1)
    return pd.DataFrame(out_data)


def main(datapath, n_samples):
    """
    Args:
        datapath (str): path to mri files
        n_samples (int): numbers of samples
    """
    dataframe = generation_coordinates(find_sample(datapath), n_samples)
    dataframe.to_csv("data/dataset.csv", index=False)
    dataframe[dataframe["split"] == 0][["images", "labels", "coords"]].to_csv(
        "data/dataset_train.csv", index=False
    )
    dataframe[dataframe["split"] == 1][["images", "labels", "coords"]].to_csv(
        "data/dataset_valid.csv", index=False
    )
    # dataframe.iloc[-1, :2].to_csv(f"data/dataset_infer.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folders to files")
    parser.add_argument("datapath", type=str, help="dir with image")
    parser.add_argument("n_samples", type=int, help="number of sample")
    params = parser.parse_args()

    main(params.datapath, params.n_samples)
