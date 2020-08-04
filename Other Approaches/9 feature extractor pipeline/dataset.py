import glob
import random
import os
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.0])
std = np.array([1.0])


class Dataset(Dataset):
    def __init__(self, dataset_path, split_path, split_number, model_flag, input_shape, training):
        self.training = training
        self.sequences, self.labels = self._extract_sequence_paths_and_labels(dataset_path, split_path, split_number, training) # creating a list of directories where the extracted frames are saved
        self.label_names = ["Non-stalled", "Stalled"] #Getting the label names or name of the class
        self.num_classes = len(self.label_names) # Getting the number of class
        self.model_flag = model_flag
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ) # This is to transform the datasets to same sizes, it's basically resizing -> converting the image to Tensor image -> then normalizing the image -> composing all the transformation in a single image


    def _extract_sequence_paths_and_labels(
        self, dataset_path, split_path="data/traintestlist", split_number=0, training=True
    ):
        """ Extracts paths to sequences given the specified train / test split """
        fn = f"fold_{split_number}_train.csv" if training else f"fold_{split_number}_test.csv"
        split_path = os.path.join(split_path, fn)
        df = pd.read_csv(split_path)
        file_name = df['filename'].values
        all_labels = df['class'].values
        sequence_paths = []
        classes = []
        for i , video_name in enumerate(file_name):
            seq_name = video_name.split(".mp4")[0]
            sequence_paths += [os.path.join(dataset_path, seq_name).replace('\\','/')]
            classes += [all_labels[i]]
        return sequence_paths, classes

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        image_path = image_path.replace('\\','/')
        try:
            return int(image_path.split('/')[-1].split('.jpg')[0])
        except:
            print("Got error while getting image number ....")
            exit()

    
    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        target = self.labels[index % len(self)]
        # Sort frame sequence based on frame number 
        image_paths = sorted(glob.glob(sequence_path+'/*.jpg'), key=lambda path: self._frame_number(path))

        

        # Pad frames of videos shorter than `self.sequence_length` to length
        image_sequence=[]

        if self.model_flag=='ALL':
            for i in range(len(image_paths)):
                img=Image.open(image_paths[i])
                image_tensor = self.transform(img)
                image_sequence.append(image_tensor)
            return image_sequence[0],image_sequence[1],image_sequence[2],image_sequence[3], \
                image_sequence[4],image_sequence[5],image_sequence[6],image_sequence[7], \
                image_sequence[8], target

        else:
            img=Image.open(image_paths[self.model_flag])
            image_tensor = self.transform(img)
            return image_tensor, target
        

    def __len__(self):
        return len(self.sequences)
