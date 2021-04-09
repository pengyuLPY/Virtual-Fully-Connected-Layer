Virtual Fully-Connected Layer: Training a Large-Scale Face Recognition Dataset with Limited Computational Resources
====


This is a PyTorch implementation of "**Virtual Fully-Connected Layer: Training a Large-Scale Face Recognition Dataset with Limited Computational Resources**" which published in CVPR2021.

## Abstract
Recently, deep face recognition has achieved significant progress because of Convolutional Neural Networks
(CNNs) and large-scale datasets. However, training CNNs
on a large-scale face recognition dataset with limited computational resources is still a challenge. This is because
the classification paradigm needs to train a fully-connected
layer as the category classifier, and its parameters will
be in the hundreds of millions if the training dataset contains millions of identities. This requires many computational resources, such as GPU memory. The metric learning paradigm is an economical computation method, but its
performance is greatly inferior to that of the classification
paradigm. To address this challenge, we propose a simple
but effective CNN layer called the Virtual fully-connected
(Virtual FC) layer to reduce the computational consumption of the classification paradigm. Without bells and whistles, the proposed Virtual FC reduces the parameters by
more than 100 times with respect to the fully-connected
layer and achieves competitive performance on mainstream
face recognition evaluation datasets. Moreover, the performance of our Virtual FC layer on the evaluation datasets
is superior to that of the metric learning paradigm by a
significant margin


## Pipeline
![image](https://github.com/pengyuLPY/Virtual-Fully-Connected-Layer/blob/master/imgs/pipeline.png)


## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 1.6.0](http://pytorch.org/)


## Usage
* set implementation details in: config.py
* run: python train.py
* extrace feature in: extract_feature.py

## ﻿Comparison with other methods

![image](https://github.com/pengyuLPY/Virtual-Fully-Connected-Layer/blob/master/imgs/comparison_sota.png)


## Bibtex

If you find the code useful, please consider citing our paper:
```
@InProceedings{Li2021VirtualFC,
author = {Li, Pengyu and Wang, Biao and Zhang, Lei},
title = {Virtual Fully-Connected Layer: Training a Large-Scale Face Recognition Dataset with Limited Computational Resources},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2021}
}
```

