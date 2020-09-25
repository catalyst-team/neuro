import argparse
import os
import joblib

import pandas as pd
import numpy as np
import nibabel as nib
import h5py



def get_labels():
    with open("./presets/label_protocol_unique.txt", "r") as f:
        t = f.read()

    labels = [int(x) for x in t.split(",")]

    return labels


def create_one_hot_voxel_labels(labels, voxel_labels, one_hot_voxel_labels_shape):
    one_hot_voxel_labels = np.array([voxel_labels == l for l in labels])
    one_hot_voxel_labels = np.pad(one_hot_voxel_labels, (
        (0,one_hot_voxel_labels_shape[0] - one_hot_voxel_labels.shape[0]),
        (0,one_hot_voxel_labels_shape[1] - one_hot_voxel_labels.shape[1]),
        (0,one_hot_voxel_labels_shape[2] - one_hot_voxel_labels.shape[2]),
        (0,one_hot_voxel_labels_shape[3] - one_hot_voxel_labels.shape[3])), mode='constant')
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


def main(datapath):
    """
    Args:
        datapath (str): path to mri files
        n_samples (int): numbers of samples
    """
    labels = get_labels()
    dataframe = find_sample(datapath)
    for i, row in dataframe.iterrows():
        voxel_labels = nib.load(row['labels']).get_fdata()
        one_hot_voxel_labels_shape = (106, 256, 256, 256)
        one_hot_voxel_labels = create_one_hot_voxel_labels(labels, voxel_labels, one_hot_voxel_labels_shape)

        with open(row['labels'].split('.nii')[0] + '.joblib', 'wb') as f:
            joblib.dump(one_hot_voxel_labels, f, compress=True)

    dataframe['one_hot_labels'] = dataframe['labels'].apply(lambda a:
                                                            a.split('.nii')[0]
                                                            + '.joblib')

    dataframe.to_csv("./data/dataset.csv", index=False)
    dataframe.iloc[:80, :].to_csv("./data/dataset_train.csv", index=False)
    dataframe.iloc[80:100, :].to_csv("./data/dataset_valid.csv", index=False)
    dataframe.iloc[-1, :].to_csv("./data/dataset_infer.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folders to files")
    parser.add_argument("datapath", type=str, help="dir with image")
    params = parser.parse_args()

    main(params.datapath)
