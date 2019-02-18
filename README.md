# DeepVisualQualityPrediction

This project is based on [Estimation of distortion sensitivity for visual quality prediction using a convolutional neural network](https://www.sciencedirect.com/science/article/pii/S1051200418308868).

## Quick start

- [Managing conda environment](#managing-conda-environment)
- [Dataset](#dataset)
- [Training](#training)
- [Status](#status)

## Managing conda environment

To create the conda environment needed to run the code on this project, run:

```shell
$ conda env create -f environment.yml
Solving environment: /
...
```

## Dataset

We've used the [LIVE dataset](http://live.ece.utexas.edu/research/quality/subjective.htm) for training and testing. You just need to download and extract it on root directory of the project.

## Training

In order to train the model, run:

```shell
$ python -Wi train.py
...
```

## Status

- [x] Prepare dataset
- [x] Create train.py
- [ ] Create test.py
