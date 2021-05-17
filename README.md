# Catalyst.Neuro [![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw) [![Github contributors](https://img.shields.io/github/contributors/catalyst-team/neuro.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/neuro/graphs/contributors)

This repository provides a 3D Brain Segmentation Pipeline for a structural
magnetic resonance imaging (MRI) scan using the Catalyst
framework using the Mindboggle Dataset as an example.  It also provides
pre-trained models for Gray Matter White Matter (GMWM) segmentation, 104 class
brain atlas segmentation, and 31 class brain atlas segmentation and with usage
shown in example
[notebooks](https://github.com/catalyst-team/neuro#example-segmentation-notebooks).
This work is based on the following papers: [An (almost) instant brain atlas
segmentation for large-scale studies](https://arxiv.org/pdf/1711.00457.pdf) and
[End-to-end learning of brain tissue segmentation from imperfect
labeling](https://arxiv.org/pdf/1612.00940.pdf).

Segmenting a structural MRI is an important processing step that enables
subsequent inferences about tissue changes in development, aging, and disease.
For our example segmentation pipeline, we use manual asotations from the
Mindboggle [dataset](https://mindboggle.readthedocs.io/en/latest/labels.html)
(DKT cortical labeling protocol).  These are considered the gold standard and
labeling a single MRI takes ~ 1 week of expert labeling.  The labeling is done
using a 2D display, one slice at a time which can lead to accuracy/ consistency
issues.  For our pre-trained models, we use automated labels from the Human
Colabectomeectome Project (HCP), which can be downloaded
[here](https://cran.r-project.org/web/packages/neurohcp/vignettes/hcp.html).
The automated labeling tool used (FreeSurfer) employs probabilistic methods
with priors to perform segmentation and is the current SOTA.  While ruinging
FreeSurfer involves more than segmentation, it can take hours to segment a
single MRI from FreeSurfer.  A MeshNet model can perform an MRI segmentation
with acceptable accuracy within minutes.

Developed in a partnership with

<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/TReNDS_logo.png)](https://trendscenter.org)

**Brain image analysis**

</div>

## Pretrained Model Statistics

| Model      | Classes | # of Brains (Training) | # of Brains (Validation) | # of Brains Test | Macro DICE |
| -----------| ------- | ----------- | ----------- | ----------- | ---------- |
| MeshNet GMWM | 3 | 20 | 4 | 4 | .9565 |
| MeshNet Dropout GMWM | 3 | 20 | 4 | 4 | .8748 |
| MeshNet Large GMWM | 3 | 20 | 4 | 4 | .9652 |
| MeshNet Large GMWM | 3 | 20 | 4 | 4 | .9624  |
| UNet GMWM | 3 | 20 | 4 | 4 | .9624 |
| MeshNet Large Mindboggle | 31 | 70 | 10 | 20 | .6742 |
| UNet Mindboggle | 31 | 70 | 10 | 20 | .6771 |
| MeshNet Large HCP Atlas | 104 | 770 | 27 | 100 | ~.85 |

| Model      | Inference Speed | Model Size |
| -----------| --------------- |----------- |
| MeshNet GMWM | 116 subvolumes/sec | .89 mb |
| MeshNet Dropout GMWM | 115 subvolumes/sec | .89 mb |
| MeshNet Large GMWM | 19 subvolumes/sec | 9mb |
| MeshNet Large Dropout GMWM | 19 subvolumes/sec | 9mb |
| UNet GMWM | 13 subvolumes/sec |  288 mb |
| MeshNet Large Mindboggle | 19 subvolumes/sec |  9 mb |
| UNet Mindboggle | 13 subvolumes/sec |  288 mb |
| MeshNet Large HCP Atlas | 18 subvolumes/sec |  10 mb |


Download links are in the Example Segmentation Notebooks

## Example Segmentation Notebooks
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/neuro/blob/master/examples/GMWM_Prediction_and_Visualization.ipynb) [Gray White Matter Prediction and Visualization](./examples/GMWM_Prediction_and_Visualization.ipynb)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/neuro/blob/master/examples/Mindboggle_Prediction_and_Visualization.ipynb) [Mindboggle Prediction and Visualization](./examples/Mindboggle_Prediction_and_Visualization.ipynb)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/neuro/blob/master/examples/Neuro_Demo.ipynb) [Neuro UNet/Meshnet training tutorial](./examples/Neuro_Demo.ipynb)

## Training MeshNet on Mindboggle

You can reproduce MeshNet for Mindboggle with 5 simple steps
- Install requirements
    ```bash
    conda env create -f neuro_conda.yml
    conda activate neuro
    pip install -r ./requirements/requirements.txt
    ```
- Download data. You need to have an account on the open science framework.
    ```bash
    mkdir Mindboggle_data
    osf -p 9ahyp clone Mindboggle_data
    cp -r Mindboggle_data/osfstorage/Mindboggle101_volumes/ data/Mindboggle_data/

    If you don't want to make an OSF account the files can also be downloaded
    the correct folder using these commands:

    download-gdrive 1l3YCRW7ezV9pw0e3aeYDFi4VsAm_DO7l data/Mindboggle_data/
    download-gdrive 1fxvqrs98F1Gnozg-HsxCjfTrw5zJbQxm data/Mindboggle_data/
    download-gdrive 1w0VXG7mLkE9tULPIxhnNK0qyguuQ7Zx8 data/Mindboggle_data/
    download-gdrive 1soDBWB0iXb3Dc1XarSKgfgwiKDInSLQw data/Mindboggle_data/
    download-gdrive 1jmwE69imokKJJaPKdCNK0Qlxb_r1bGN4 data/Mindboggle_data/

- Unzip data and remove archive files
    ```bash
    find data/Mindboggle_101 -name '*.tar.gz'| xargs -i tar zxvf {} -C data/Mindboggle_101
    find data/Mindboggle_101 -name '*.tar.gz'| xargs -i rm {}
    ```
- Prepare data (31 is the number of classes in the example. 102 is the full
  number of classes in Mindboggle)
    ```bash
    python neuro/scripts/prepare_data.py ./data/Mindboggle_101 31
    ```
- Start training
    ```bash
    python neuro/minimal_example.py
    ```

## HCP Data preparation
- Prepare **T1 or T2 input** with mri_convert from [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
to conform T1 to 1mm voxel size in coronal slice direction with side length 256.
**You can skip this step if your T1 image is already with slice thickness 1mm x 1mm x 1mm and 256 x 256 x 256.**
```
mri_convert *brainDir*/t1.nii *brainDir*/T1.nii.gz -c
```
- Prepare **104 atlas labels** from aparc+aseg.nii.gz for HCP atlas
  segmentation using:
```
python neuro/scripts/prepare_atlas_data.py --brains_list *brains_list.txt*
```
- Or Prepare **GMWM labels** from aparc+aseg.mgz using:
```
./bin/mk_gwmwm_labels.sh [input_directory] [output_directory]
```

----

<div align="center">


[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated Deep Learning R&D**

[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-on%20twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)

</div>
