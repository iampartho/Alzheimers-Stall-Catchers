import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class VoxelDataset(Dataset):
    def __init__(self, dataset_path, split_path, split_number, input_shape, training):
        self.training = training
        self.sequences, self.labels = self._extract_sequence_paths_and_labels(dataset_path, split_path, split_number,
                                                                              training)  # creating a list of directories where the extracted frames are saved
        self.label_names = ["Non-stalled", "Stalled"]  # Getting the label names or name of the class
        self.num_classes = len(self.label_names)  # Getting the number of class
        self.input_shape = input_shape

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

    def pc2voxel(self, cloud0, cloud1, cloud2, depth=32, height=64, width=64):

        voxel_grid = np.zeros((3, depth, height, width), dtype=np.float16)

        in_depth = max(np.max(cloud0[:, 0]), np.max(cloud1[:, 0]), np.max(cloud2[:, 0]))
        in_height = max(np.max(cloud0[:, 1]), np.max(cloud1[:, 1]), np.max(cloud2[:, 1]))
        in_width = max(np.max(cloud0[:, 2]), np.max(cloud1[:, 2]), np.max(cloud2[:, 2]))

        if in_depth >= depth:
            depth_ratio = depth / (in_depth + 1)
            cloud0[:, 0] = np.uint32(cloud0[:, 0].astype(float) * depth_ratio)
            cloud1[:, 0] = np.uint32(cloud1[:, 0].astype(float) * depth_ratio)
            cloud2[:, 0] = np.uint32(cloud2[:, 0].astype(float) * depth_ratio)
        if in_height >= height:
            height_ratio = height / (in_height + 1)
            cloud0[:, 1] = np.uint32(cloud0[:, 1].astype(float) * height_ratio)
            cloud1[:, 1] = np.uint32(cloud1[:, 1].astype(float) * height_ratio)
            cloud2[:, 1] = np.uint32(cloud2[:, 1].astype(float) * height_ratio)
        if in_width >= width:
            width_ratio = width / (in_width + 1)
            cloud0[:, 2] = np.uint32(cloud0[:, 2].astype(float) * width_ratio)
            cloud1[:, 2] = np.uint32(cloud1[:, 2].astype(float) * width_ratio)
            cloud2[:, 2] = np.uint32(cloud2[:, 2].astype(float) * width_ratio)

        voxel_grid[0, cloud0[:, 0], cloud0[:, 1], cloud0[:, 2]] = 1.0
        voxel_grid[1, cloud1[:, 0], cloud1[:, 1], cloud1[:, 2]] = 1.0
        voxel_grid[2, cloud2[:, 0], cloud2[:, 1], cloud2[:, 2]] = 1.0

        return voxel_grid

    def get_cloud(self, filename):
        depth = self.input_shape[0]
        height = self.input_shape[1]
        width = self.input_shape[2]

        hf = h5py.File(filename, 'r')
        c1 = hf['cloud1'][:]
        c2 = hf['cloud2'][:]
        c3 = hf['cloud3'][:]
        hf.close()

        X = self.pc2voxel(c1, c2, c3, depth=depth, height=height, width=width)
        X = torch.from_numpy(X).float()
        return X

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        target = self.labels[index % len(self)]

        voxels = self.get_cloud(sequence_path + ".h5")

        return voxels, target





batch_size = 14

dataset_path = 'data/point_cloud'
split_path = 'traintestlist'
split_number = 0
num_epochs = 50
checkpoint_model = ''

voxel_shape = [32, 64, 64]



# Define training set
train_dataset_vox = VoxelDataset(
    dataset_path=dataset_path,
    split_path=split_path,
    split_number=split_number,
    input_shape=voxel_shape,
    training=True,
)
train_dataloader_vox = DataLoader(train_dataset_vox, batch_size= batch_size,sampler=BalancedBatchSampler(train_dataset_vox),shuffle=False, num_workers=4)
# Define test set
test_dataset_vox = VoxelDataset(
    dataset_path=dataset_path,
    split_path=split_path,
    split_number=split_number,
    input_shape=voxel_shape,
    training=False,
)
test_dataloader_vox = DataLoader(test_dataset_vox, batch_size=batch_size, shuffle=False, num_workers=4)