import glob
import random
import os
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0])
std = np.array([1])


class ImageDataset(Dataset):
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length, training):
        self.training = training
        self.sequences, self.labels = self._extract_sequence_paths_and_labels(dataset_path, split_path, split_number,
                                                                              training)  # creating a list of directories where the extracted frames are saved
        self.sequence_length = int(
            sequence_length)  # Defining how many frames should be taken per video for training and testing
        self.label_names = ["Non-stalled", "Stalled"]  # Getting the label names or name of the class
        self.num_classes = len(self.label_names)  # Getting the number of class
        self.input_shape = input_shape
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )  # This is to transform the datasets to same sizes, it's basically resizing -> converting the image to Tensor image -> then normalizing the image -> composing all the transformation in a single image

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
        for i, video_name in enumerate(file_name):
            seq_name = video_name.split(".mp4")[0]
            sequence_paths += [os.path.join(dataset_path, seq_name).replace('\\', '/')]
            classes += [all_labels[i]]
        return sequence_paths, classes

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        image_path = image_path.replace('\\', '/')
        try:
            return int(image_path.split('/')[-1].split('.jpg')[0])
        except:
            print("Got error while getting image number ....")
            exit()

    def _pad_to_length(self, sequence, path):
        """ Pads the video frames to the required sequence length for small videos"""
        try:
            left_pad = sequence[0]
        except:
            print("Got error while padding ....")
            exit()
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        target = self.labels[index % len(self)]
        # Sort frame sequence based on frame number 
        image_paths = sorted(glob.glob(sequence_path + '/*.jpg'), key=lambda path: self._frame_number(path))

        # Pad frames of videos shorter than `self.sequence_length` to length

        image_paths = self._pad_to_length(image_paths, sequence_path)
        total_image = len(image_paths)
        if total_image >= self.sequence_length and total_image < (
                self.sequence_length + int(self.sequence_length // 2)):
            midpoint = (total_image // 2)
            sample_interval = 1
            start_i = (midpoint - (self.sequence_length // 2))
            end_i = start_i + self.sequence_length
        elif total_image >= (self.sequence_length + int(self.sequence_length // 2)):
            midpoint = (total_image // 2)
            sample_interval = 1
            start_i = (midpoint - (self.sequence_length // 2) + int((self.sequence_length // 2) // 2)) - 1
            end_i = start_i + self.sequence_length
        else:
            start_i = 0
            end_i = total_image
            sample_interval = 1
        # flip = np.random.random() < 0.5
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, end_i, sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                img = Image.open(image_paths[i])
                image_tensor = self.transform(img)
                # if flip:
                #     image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        image_sequence = torch.stack(image_sequence)
        image_sequence = image_sequence.view(3, self.sequence_length, self.input_shape[-2], self.input_shape[-2])
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)






batch_size = 14

dataset_path = 'data/micro_frames'
split_path = 'traintestlist'
split_number = 0
sequence_length=40
num_epochs = 50
img_dim = 112
channels = 3
latent_dim = 512
checkpoint_model = ''

image_shape = (channels, img_dim, img_dim)



# Define training set
train_dataset_img = ImageDataset(
    dataset_path=dataset_path,
    split_path=split_path,
    sequence_length=sequence_length,
    split_number=split_number,
    input_shape=image_shape,
    training=True,
)
train_dataloader_img = DataLoader(train_dataset_img, batch_size= batch_size,sampler=BalancedBatchSampler(train_dataset_img),shuffle=False, num_workers=4)
# Define test set
test_dataset_img = ImageDataset(
    dataset_path=dataset_path,
    split_path=split_path,
    split_number=split_number,
    sequence_length=sequence_length,
    input_shape=image_shape,
    training=False,
)
test_dataloader_img = DataLoader(test_dataset_img, batch_size=batch_size, shuffle=False, num_workers=4)