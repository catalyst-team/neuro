#!/bin/bash
DATADIR=$1
OUTDIR=$2


CURDIR=`pwd`
cd $DATADIR

if [ ! -f aparc+aseg.nii ]; then
   mri_convert aparc+aseg.mgz ${OUTDIR}/aparc+aseg.nii.gz
fi

if [ ! -f T1.nii ]; then
   mri_convert T1.mgz ${OUTDIR}/T1.nii.gz
fi

cd $OUTDIR

if [ ! -f all_wmN.nii ]; then
   3dcalc -a aparc+aseg.nii.gz -expr 'equals(a,2)+equals(a,41)+equals(a,7)+equals(a,16)+equals(a,46)+and(step(a-250),step(256-a))' -prefix all_wmN.nii
fi

if [ ! -f all_gmN.nii ]; then
   3dcalc -a aparc+aseg.nii.gz -expr 'and(step(a-1000),step(1036-a))+and(step(a-2000),step(2036-a))+and(step(a-7),step(14-a))+and(step(a-16),step(21-a))+and(step(a-25),step(29-a))+and(step(a-46),step(56-a))+and(step(a-57),step(61-a))' -prefix all_gmN.nii
fi

if [ ! -f labels.nii.gz ]; then
	3dcalc -a all_gmN.nii -b all_wmN.nii -expr 'a+2*b' -prefix labels.nii.gz
fi

python << END
from nipy import save_image, load_image
import numpy as np
T1 = load_image('${OUTDIR}/T1.nii.gz')
labels = load_image('${OUTDIR}/labels.nii.gz')
np.save('${OUTDIR}/affine.npy',T1.affine)
np.save('${OUTDIR}/T1.npy', T1.get_data())
np.save('${OUTDIR}/labels.npy', labels.get_data())
END

rm -rf ${OUTDIR}/all_wmN.nii
rm -rf ${OUTDIR}/all_gmN.nii
rm -rf ${OURDIR}/labels.nii.gz

cd $CURDIR
