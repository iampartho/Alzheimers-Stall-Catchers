# This Readme explains the differenct sections of the "[training_notebook](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/training_notebook.ipynb)"

## Checking the GPU name
Since most of our training were conducted in mixed precision we try to use the Google Colab's Tesla T4 GPU therefore we always checked the GPU name at the very begining of the training.
## Installing Apex
For mixed precision training [Apex](https://developer.nvidia.com/blog/apex-pytorch-easy-mixed-precision-training/) is the most reliable and most used library. But using Apex everytime we want to train is rather painful as each time we need to install the packages in Google colab. So the later training we used a manual kind of way to do the exact same thing the package does while conducting a mix-precision training.

## Mounting Google Drive

The dataset were uploaded to the google drive and we fetched the data from our drive to the colab's workspace each time we conduct a training. Before doing that we first needed to mount the drive to colab's workspace.

## Setting up the workspace

This basically fetch all the data from google drive to colab's workspace. These data contains the training and validation folds that we create for training and also all images frames and as well as weight files.(As we did not use anything else than colab so we need to conduct the training of a single model in a multiple phase)

## Dataset.py

This is the intial dataset preperation code we used. From the video we extract just the region of interest portion and saved them as the grayscale image frame using [this](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/extract_frames.py) code. In here we used a grayscale frame read in a three channel(in order to use the provided mean and standard deviation array) and resize them and at the end normalised them. We sampled 40 frames from a video, the frames were sampled in the following manner , 
* First we find the mid-frame of the video
* Then go back 10 frmaes from there
* And sampled 40 frames from that point
We took two sets of 40 frames, one is going in forward manner and other is going in backward manner. The dataset will return two set of 40 frames(one in forward manner and another in backward manner) 

**Following is sample of a single image frame which was primarily used to train the network**

<center><img src="https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/51.jpg" /></center>

![Grayscle image](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/51.jpg)

## Dataset with rotation

The cell of code is used to train our final model. The data used here is the RGB data rather than the previous grayscale data. The code used to extract these frames is [this.](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/extract_frames_new.py) The basic difference between the previous data and this data is that we saw that after croping out the region of interest from each video frame there were many pixels (which were basically don't care pixels) which were black. And since black is one of a significant colour for our classification so therefore we coloured these don't care pixel as raw blue to distinguish it from the actual black pixel inside the region of interest. This hypothesis of ours seems to give a great jump in the leaderboard. Dataset preperation is quite similar as we discussed in the previous section. The one basic change is the added rotation of 7 particular degrees with a probability of 0.5. 

**Following is a sample of RGB image frame**

![RGB image](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/50.jpg)

## Model.py

This cell is the model architecture for backbone "[resnet(2+1)d](https://arxiv.org/abs/1711.11248)". 
**Following figure shows the workflow of the architecture**
![Model Architecture](https://github.com/iampartho/Alzheimers-Stall-Catchers/blob/master/Image%20Based%20Approach/Model%20architecture.jpg)

## Model (Big architecture)

This cell is the model architecture for backbone "[Resnext-101](https://arxiv.org/abs/1711.09577)".

## Balance Batch

This cell of code create a balance batch by over-sampling in per mini-batch. It was an integral part of our multi-phase training.

## Ranger optimizer

The cell is used to use [Ranger optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer). This optimizer is used to train the models to get further boost in the **MCC**.

## Training
This cell of code is used to train the algorithm. Since we used colab to train all of our model therefore we have to train for multiple phases. Some of the key points in our training were,
* Balance-batch
* Learning Rate = 0.0001 (No scheduling)
* Adam and Ranger optimizer used in a sequential way
* Batch-size = 8
* Used gradiend accumulation technique with an accumulation step of 8
* Optimized for best validation MCC
* Training and validation set both have a 70:30 non-stall to stall instance distribution 

