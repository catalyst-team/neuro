import argparse
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser(description='Prepare ATLAS anatomical ground truth')
parser.add_argument('--brains_list')
args = parser.parse_args()

labels = [i for i in range(1001,1004)] + [i for i in range(1005,1036)] + [i for i in range(2001,2004)] + [i for i in range(2005, 2036)] + [10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58, 28, 60, 2, 41, 4, 5, 43, 44, 14, 15, 24, 16, 7, 46, 8, 47, 251, 252, 253, 254, 255]
print (len(labels), labels)
filename = 'aparc+aseg.nii.gz'
output_filename = 'atlas_full_104.nii.gz'
f = open(args.brains_list, 'r')
brains = f.read().splitlines()
for i, b in enumerate(brains):
    print (b)
    img = nib.load(b + filename).get_fdata()
    segmentation = np.zeros(img.shape)
    k = 1
    for l in labels:
        mask = np.equal(l, img)
        segmentation[mask] = k
        k += 1
    if (len(np.unique(segmentation)) == (len(labels) + 1)):
        nib_seg = nib.Nifti1Image(segmentation, np.eye(4))
        nib_seg.to_filename(b + output_filename)

    else:
        assert False, 'Bad {}'.format(b)
