# This Readme explains the differenct sections of the "[training_testing.ipynb](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/training_testing.ipynb)"


## Video Frame Dataset Generator

This is a dataset generator class build on top of the pytoch Dataset class. In this code we read the image frames from files for a particular video and stack those frames into two tensor, one is going forward and other is going backward manner.

**Following is sample of a single image frame which was primarily used to train the network**


![Preprocessed image](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/29.jpg)


## ## Model

This cell defines the encoder with it's forward propagation function. The model used for our experiment was "[resnet(2+1)d](https://arxiv.org/abs/1711.11248)". We trained this model with out preprocessed data.



## Training + Validation
This cell of code is used to train the algorithm and validate the encoder. Some of the key points in our training were,
* Batch size = 1
* Learning Rate = 0.0001 (OneCycleLR Scheduling)
* Adam optimizer
* Used gradiend accumulation technique with an accumulation step of 32
* Optimized for best validation MCC
* Training and validation set both have a 50:50 non-stall to stall instance distribution 

