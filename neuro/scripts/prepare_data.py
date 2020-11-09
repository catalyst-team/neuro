import argparse
import os

import h5py
import joblib
import nibabel as nib
import numpy as np
import pandas as pd


def get_labels():
    with open("./presets/label_protocol_unique.txt", "r") as f:
        t = f.read()

    labels = [int(x) for x in t.split(",")]

    return labels


def create_one_hot_voxel_labels(
    labels, voxel_labels, one_hot_voxel_labels_shape
):
    one_hot_voxel_labels = np.array([voxel_labels == l for l in labels])
    one_hot_voxel_labels = np.pad(
        one_hot_voxel_labels,
        (
            (0, one_hot_voxel_labels_shape[0] - one_hot_voxel_labels.shape[0]),
            (0, one_hot_voxel_labels_shape[1] - one_hot_voxel_labels.shape[1]),
            (0, one_hot_voxel_labels_shape[2] - one_hot_voxel_labels.shape[2]),
            (0, one_hot_voxel_labels_shape[3] - one_hot_voxel_labels.shape[3]),
        ),
        mode="constant",
    )
    return one_hot_voxel_labels.astype("uint8")


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
            for train in os.listdir(person_folder):
                if train.startswith("."):
                    continue
                if train == "t1weighted.nii.gz":
                    labels_data["images"].append(
                        os.path.join(person_folder, "t1weighted.nii.gz")
                    )
                if train == "labels.DKT31.manual+aseg.nii.gz":
                    labels_data["labels"].append(
                        os.path.join(
                            person_folder, "labels.DKT31.manual+aseg.nii.gz"
                        )
                    )
                    print(t)
                    t += 1

    return pd.DataFrame(labels_data)


def main(datapath, n_labels):
    """
    Args:
        datapath (str): path to mri files
        n_labels (int): number of labels generated (the n most frequent labels)
    """
    labels = [1002, 1003]
    labels.extend([*range(1005, 1032)])
    labels.extend([1035, 1036])
    print(len(labels))
    # labels = [i for i in range(1001, 1004)] + [i for i in range(1005, 1036)] + \
    #         [i for i in range(2001, 2004)] + [i for i in range(2005, 2036)] + \
    #         [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58, 28, 60,
    #          2, 41, 4, 5, 43, 44, 14, 15, 24, 16, 7, 46, 8, 47, 251, 252, 253,
    #          254, 255]

    dataframe = find_sample(datapath)
    df_list = []
    for i, row in dataframe.iterrows():
        voxel_labels = nib.load(row["labels"]).get_fdata()
        unique, counts = np.unique(voxel_labels, return_counts=True)
        temp_df = pd.DataFrame({"labels": unique, "counts": counts}).T
        new_df = pd.DataFrame(temp_df.values[1:], columns=temp_df.iloc[0])
        df_list.append(new_df)

    full_value_counts = pd.concat(df_list)
    # labels = [l for l in full_value_counts.sum().nlargest(n_labels).index.tolist() if l != 0]

    for i, row in dataframe.iterrows():
        voxel_labels = nib.load(row["labels"]).get_fdata()
        new_img = np.zeros([256, 256, 256])
        new_img[
            : voxel_labels.shape[0],
            : voxel_labels.shape[1],
            : voxel_labels.shape[2],
        ] = voxel_labels
        segmentation = np.zeros([256, 256, 256])

        for i, l in enumerate(labels):
            mask = np.equal(l, new_img)
            segmentation[mask] = i
        nib_seg = nib.Nifti1Image(segmentation, np.eye(4))
        nib_seg.to_filename(
            row["labels"].split(".nii")[0] + "_labels" + ".nii.gz"
        )

    dataframe["nii_labels"] = dataframe["labels"].apply(
        lambda a: a.split(".nii")[0] + "_labels" + ".nii.gz"
    )

    dataframe.to_csv("./data/dataset.csv", index=False)
    dataframe.iloc[:80, :].to_csv("./data/dataset_train.csv", index=False)
    dataframe.iloc[80:100, :].to_csv("./data/dataset_valid.csv", index=False)
    dataframe.iloc[-1, :].to_csv("./data/dataset_infer.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folders to files")
    parser.add_argument("datapath", type=str, help="dir with image")
    parser.add_argument("n_labels", type=int, help="dir with image")
    params = parser.parse_args()

    main(params.datapath, params.n_labels)
