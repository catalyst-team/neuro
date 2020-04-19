from infer import Predictor
import argparse
import numpy as np
import nibabel as nib
import os

def Brain_Segmenatation(path_in, path_out, path_model):
    mask_predictor = Predictor(path_model)
    probability = mask_predictor.predict(path_in)
    ni_img = nib.Nifti1Image(probability, affine=np.eye(4))
    nib.save(ni_img, os.path.join(path_out,'output.nii.gz'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'folders to files')
    parser.add_argument('path_in', type=str, help='dir with image')
    parser.add_argument('path_out', type=str, help='output dir')
    parser.add_argument('path_model', type = str, help = 'path_model')
    args = parser.parse_args()
    Brain_Segmenatation(args.path_in, args.path_out, args.path_model)