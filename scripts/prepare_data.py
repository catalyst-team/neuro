import pathlib
import pandas as pd
import os

def find_sample(path):
    labels_data = {"images":[],"labels":[]}
    for case in os.listdir(path):
        case_folder = os.path.join(path, case)
        for person in os.listdir(case_folder):
            if person in ["._.DS_Store",".DS_Store"]:
                continue
            person_folder = os.path.join(case_folder, person)
            for train in os.listdir(person_folder):
                if train == 't1weighted.nii':
                    labels_data["images"].append(os.path.join(person_folder,'t1weighted.nii'))
                if train == 'labels.DKT31.manual+aseg.nii':
                    labels_data["labels"].append(os.path.join(person_folder,'labels.DKT31.manual.nii'))
    return pd.DataFrame(labels_data)





def main(
    datapath: str = "../data/dataverse_files", valid_size: float = 0.2, random_state: int = 42
):
    dataframe = find_sample(datapath)
    dataframe.to_csv(f"data/dataset.csv", index=False)
if __name__ == "__main__":
    main()
