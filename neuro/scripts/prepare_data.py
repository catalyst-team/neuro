import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd


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
    # Labels are from https://mindboggle.readthedocs.io/en/latest/labels.html
    # DKT protocol minus removed labels
    labels = [1002, 1003]
    labels.extend([*range(1005, 1032)])
    labels.extend([1035, 1036])
    labels.extend([2002, 2003])
    labels.extend([*range(2005, 2032)])
    labels.extend([2034, 2035])

    # Non-cortical labels
    labels.extend(
        [
            16,
            24,
            14,
            15,
            72,
            85,
            4,
            5,
            6,
            7,
            10,
            11,
            12,
            13,
            17,
            18,
            25,
            26,
            28,
            30,
            91,
            43,
            44,
            45,
            46,
            49,
            50,
            51,
            52,
            53,
            54,
            57,
            58,
            60,
            62,
            92,
            630,
            631,
            632,
        ]
    )

    n_labels = min(n_labels, len(labels))
    labels = labels[:n_labels]

    dataframe = find_sample(datapath)
    df_list = []
    for i, row in dataframe.iterrows():
        voxel_labels = nib.load(row["labels"]).get_fdata()
        unique, counts = np.unique(voxel_labels, return_counts=True)
        temp_df = pd.DataFrame({"labels": unique, "counts": counts}).T
        new_df = pd.DataFrame(temp_df.values[1:], columns=temp_df.iloc[0])
        df_list.append(new_df)

    full_value_counts = pd.concat(df_list)

    for i, row in dataframe.iterrows():
        voxel_labels = nib.load(row["labels"]).get_fdata()
        new_img = np.zeros([256, 256, 256])
        new_img[
            : voxel_labels.shape[0],
            : voxel_labels.shape[1],
            : voxel_labels.shape[2],
        ] = voxel_labels
        segmentation = np.zeros([256, 256, 256])

        for j, l in enumerate(labels):
            mask = np.equal(l, new_img)
            segmentation[mask] = j + 1
        nib_seg = nib.Nifti1Image(segmentation, np.eye(4))
        nib_seg.to_filename(
            row["labels"].split(".nii")[0] + "_labels" + ".nii.gz"
        )

    dataframe["nii_labels"] = dataframe["labels"].apply(
        lambda a: a.split(".nii")[0] + "_labels" + ".nii.gz"
    )

    dataframe.to_csv("./data/dataset.csv", index=False)
    dataframe = dataframe.sample(frac=1, random_state=42)
    dataframe.iloc[:70, :].to_csv("./data/dataset_train.csv", index=False)
    dataframe.iloc[70:80, :].to_csv("./data/dataset_valid.csv", index=False)
    dataframe.iloc[80:, :].to_csv("./data/dataset_infer.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folders to files")
    parser.add_argument("datapath", type=str, help="dir with image")
    parser.add_argument(
        "n_labels",
        type=int,
        help="""number of labels used for segmentation.
                        The first 62 follow the DKT human labeling protocol while the next 39 are from Freesurfer""",
    )
    params = parser.parse_args()

    main(params.datapath, params.n_labels)
