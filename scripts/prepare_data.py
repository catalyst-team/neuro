import pathlib
import pandas as pd
from sklearn import model_selection
import os

def find_sample(path):
    labels_data = {"images":[],"labels":[]}
    for case in os.listdir(path):
        case_folder = os.path.join(path, case)
        for person in os.listdir(case_folder):
            person_folder = os.path.join(case_folder, person)
            try:
                for train in os.listdir(person_folder):
                    if train == 't1weighted_brain.MNI152.nii.gz':
                        labels_data["image"].append(os.path.join(person_folder,train))
                    if train == 'labels.DKT31.manual.MNI152.nii.gz':
                        labels_data["label"].append(os.path.join(person_folder,train))
            except:
                continue
    return pd.DataFrame(labels_data)





def main(
    datapath: str = "./data/Mindbonggle_101", valid_size: float = 0.2, random_state: int = 42
):
    dataframe = find_sample(datapath)

    df_train, df_valid = model_selection.train_test_split(
        dataframe,
        test_size=valid_size,
        random_state=random_state,
        shuffle=False
    )
    for source, mode in zip((df_train, df_valid), ("train", "valid")):
        source.to_csv(f"data/dataset_{mode}.csv", index=False)


if __name__ == "__main__":
    main()
