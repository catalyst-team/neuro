import pathlib
import pandas as pd
from sklearn import model_selection
import os
import shutil
import gzip

def find_sample(path):
    labels_data = {"images":[],"labels":[]}
    for case in os.listdir(path):
        case_folder = os.path.join(path, case)
        if case.endswith(".tar.gz"):
            shutil.unpack_archive(os.path.join(path, case), path)
            os.remove(os.path.join(path, case))
        for person in os.listdir(case_folder):
            if person in ["._.DS_Store",".DS_Store"]:
                continue
            person_folder = os.path.join(case_folder, person)
            for train in os.listdir(person_folder): 
                if train == 't1weighted_brain.MNI152.nii.gz':
                    with gzip.open(os.path.join(person_folder,'t1weighted_brain.MNI152.nii.gz'), "rb") as f_in:
                        with open(os.path.join(person_folder,'t1weighted_brain.MNI152.nii'), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    labels_data["images"].append(os.path.join(person_folder,'t1weighted_brain.MNI152.nii'))
                if train == 'labels.DKT31.manual.MNI152.nii.gz':
                    with gzip.open(os.path.join(person_folder,'labels.DKT31.manual.MNI152.nii.gz'), "rb") as f_in:
                        with open(os.path.join(person_folder,'labels.DKT31.manual.MNI152.nii'), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    labels_data["labels"].append(os.path.join(person_folder,'labels.DKT31.manual.MNI152.nii'))
    return pd.DataFrame(labels_data)





def main(
    datapath: str = "/home/Bekovmi/neuro-1/scripts/data/Mindbonggle_101", valid_size: float = 0.2, random_state: int = 42
):
    dataframe = find_sample(datapath)

    dataframe.to_csv(f"data/dataset.csv", index=False)


if __name__ == "__main__":
    main()
