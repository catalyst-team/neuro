import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd


def save_segmentation(filename):
    """Docs."""
    with open("../presets/label_protocol_unique.txt", "r") as f:
        t = f.read()

    labels = [int(x) for x in t.split(",")]
    path = os.path.dirname(filename)
    img = nib.load(filename, mmap=False)
    img = img.get_fdata(dtype=np.float32)
    segmentation = np.zeros([len(labels), 256, 256, 256])
    k = 0
    for l in labels:
        segmentation[k, : img.shape[0], : img.shape[1], : img.shape[2]] = (
            img == l
        )
        k += 1
    data = segmentation.astype("uint8")
    np.save(os.path.join(path, "prepared_labels.DKT31.manual.npy"), data)
    return data.tobytes()


def save_image(image_name, path):
    """Docs."""
    img = nib.load(image_name)
    img = img.get_fdata(dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255.0
    new_img = np.zeros([1, 256, 256, 256])
    new_img[
        0, : img.shape[0], : img.shape[1], : img.shape[2]
    ] = img
    np.save(os.path.join(path, "prepared_labels.DKT31.manual.npy"), new_img)
    return new_img.tobytes()


def find_sample(path):
    """Docs."""
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
                    labels_data["images"].append(
                        save_image(os.path.join(person_folder, sample), path)
                    )
                if sample == "labels.DKT31.manual+aseg.nii":
                    labels_data["labels"].append(
                        save_segmentation(os.path.join(person_folder, sample))
                    )
                    print(t)
                    t += 1

    return pd.DataFrame(labels_data)


def main(datapath):
    """Docs."""
    dataframe = find_sample(datapath)
    dataframe.to_csv("data/dataset.csv", index=False)
    dataframe.iloc[:80, :].to_csv("data/dataset_train.csv", index=False)
    dataframe.iloc[80:100, :].to_csv("data/dataset_valid.csv", index=False)
    dataframe.iloc[-1, :].to_csv("data/dataset_infer.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="folders to files")
    parser.add_argument("datapath", type=str, help="dir with image")
    params = parser.parse_args()

    main(params.datapath)

