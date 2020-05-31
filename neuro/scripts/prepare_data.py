import argparse
import os

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


def main(datapath):
    """
    Args:
        datapath (str): path to mri files
        n_samples (int): numbers of samples
    """
    dataframe = find_sample(datapath)
    dataframe.to_csv("./data/dataset.csv", index=False)
    dataframe.iloc[:80, :].to_csv("./data/dataset_train.csv", index=False)
    dataframe.iloc[80:100, :].to_csv("./data/dataset_valid.csv", index=False)
    dataframe.iloc[-1, :].to_csv("./data/dataset_infer.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folders to files")
    parser.add_argument("datapath", type=str, help="dir with image")
    params = parser.parse_args()

    main(params.datapath)
