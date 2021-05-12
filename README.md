<div align="center">


[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated DL R&D**

[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Catalyst_Deploy/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Catalyst&tab=projectOverview&guest=1)
[![CodeFactor](https://www.codefactor.io/repository/github/catalyst-team/catalyst/badge)](https://www.codefactor.io/repository/github/catalyst-team/catalyst)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-on%20twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)

</div>

PyTorch framework for Deep Learning research and development.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop. <br/>
Break the cycle - use the Catalyst!

Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md). Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing):
- [Alchemy](https://github.com/catalyst-team/alchemy) - Experiments logging & visualization
- [Catalyst](https://github.com/catalyst-team/catalyst) - Accelerated Deep Learning Research and Development
- [Reaction](https://github.com/catalyst-team/reaction) - Convenient Deep Learning models serving

[Catalyst at AI Landscape](https://landscape.lfai.foundation/selected=catalyst).

---

# Catalyst.Neuro [![Build Status](https://travis-ci.com/catalyst-team/neuro.svg?branch=master)](https://travis-ci.com/catalyst-team/neuro) [![Github contributors](https://img.shields.io/github/contributors/catalyst-team/neuro.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/neuro/graphs/contributors)


This repository provides an 3D Brain Segmentation Pipeline using the Catalyst
framework along with pretrained models and example usage.

Developed in a partnership with

<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/TReNDS_logo.png)](https://trendscenter.org)

**Brain image analysis**

</div>

## Pretrained Models

| Model      | Macro DICE |  Inference Speed | Model Size | Classes
| -----------| ----------- |----------- |----------- |----------- |
| MeshNet     | .9565 | 116 subvolumes/sec | .89 mb | 3
| MeshNet Dropout  | .8748 | 115 subvolumes/sec | .89 mb | 3
| MeshNet Large  | .9652 | 19 subvolumes/sec | 9mb | 3
| MeshNet Large Dropout  | .9354 | 19 subvolumes/sec | 9mb | 3
| UNet  | .9624  | 13 subvolumes/sec |  288 mb | 3
| MeshNet Large | .6742 | 19 subvolumes/sec |  9 mb | 31
| UNet  | .6771 | 13 subvolumes/sec |  288 mb | 31
| MeshNet Large | ~.85 | 18 subvolumes/sec |  10 mb | 104

Download links are in the Example Segmentation Notebooks

## Example Segmentation Notebooks
* Gray White Matter Prediction and Visualization
* Mindboggle Prediction and Visualization

## Training MeshNet on Mindboggle

You can reproduce MeshNet for Mindboggle with 4 simple steps
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
    find data/Mindboggle_101 -name '*.tar.gz'| xargs -i tar zxvf {} -C data/Mindboggle_101
    find data/Mindboggle_101 -name '*.tar.gz'| xargs -i rm {}
    ```
- Prepare data (31 is the number of classes in the example. 102 is the max)
    ```bash
    python neuro/scripts/prepare_data.py ./data/Mindboggle_101 31
    ```
- Start training
    ```bash
    python training/minimal_example.py
