# DeepVisualQualityPrediction

Open source implementation of the publication titled [A multi-stream network with different receptive fields to assess visual quality](https://www.researchgate.net/publication/336349460_A_multi-stream_network_with_different_receptive_fields_to_assess_visual_quality) by Luan A. Gonçalves, Ronaldo F. Zampolo and Fabrício B. Barros at [Symposium on Knowledge Discovery, Mining and Learning (KDMiLe)](http://sbbd.org.br/kdmile2019/) 2019.

## Quick start

- [Managing conda environment](#managing-conda-environment)
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
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

With you're running the code for the first time you'll need to enable the [train.py](https://github.com/LuanAGoncalves/DeepVisualQualityPrediction/blob/master/train.py) script to create dataset files. To do it, run:

```shell
$ python train.py --input=ref --network=default --visdom=0 --epochs=50 --generate=True
$ python train.py --input=dist --network=default --visdom=0 --epochs=50 --generate=False
$ python train.py --input=ref --network=MultiscaleDQP --visdom=0 --epochs=50 --generate=False
$ python train.py --input=dist --network=MultiscaleDQP --visdom=0 --epochs=50 --generate=False
```

## Testing

In order to test the models, run:

```shell
$ python DQPTest.py --input=<ref od dist> --folder=<models folder> --network=<Default or MultiscaleDQP>
```

## Status

- [x] Reproduce the results of [Estimation of distortion sensitivity for visual quality prediction using a convolutional neural network](https://www.sciencedirect.com/science/article/pii/S1051200418308868)
- [x] Use a network with different receptive fields.
