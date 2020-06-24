import csv
import glob
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm

from preprocess_images import VideoProcessor, ImageProcessor3D


class Generator:

    def __init__(self):
        self.data = []

    def extract_name(self, filename, path, suffix):
        # extracts exact filename from path
        name = filename.replace(path, "")
        name = name.replace(suffix, "")
        return name

    def generate_processed_dataset(self, src_path, dst_path, testing=False):
        src_suffix = ".mp4"
        dst_suffix = ".jpg"

        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for f in tqdm(files):
            # Iterates through all the video files and export images
            sub_directory = dst_path + "/" + self.extract_name(f, src_path, src_suffix)
            if not os.path.exists(sub_directory):   # Create a directory for each video file
                os.mkdir(sub_directory)

            extractor = VideoProcessor(f)
            processed_images = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

            for frame_no in range(processed_images.shape[0]):
                output_filename = sub_directory + "/" + str(frame_no) + dst_suffix
                cv2.imwrite(output_filename, processed_images[frame_no, :, :])

            if testing == True:
                break

    def generate_point_cloud_dataset(self, src_path, dst_path, testing = False):
        src_suffix = ".mp4"
        dst_suffix = ".h5"

        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for f in tqdm(files):
            # Iterates through all the video files and export images to file

            extractor = VideoProcessor(f)
            image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

            thresh1 = int(np.percentile(image_collection.ravel(), 99))
            thresh2 = int(np.percentile(image_collection.ravel(), 95))
            thresh3 = int(np.percentile(image_collection.ravel(), 90))

            cloud1, nil = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh1,
                                                                           filter_outliers=True)
            cloud2, nil = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh2,
                                                                           filter_outliers=True)
            cloud3, nil = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh3,
                                                                           filter_outliers=True)

            sample_name = self.extract_name(f, src_path, src_suffix)

            output_filename = dst_path + "/" + sample_name + dst_suffix
            hf = h5py.File(output_filename, 'w')
            hf.create_dataset('cloud1', data=cloud1)
            hf.create_dataset('cloud2', data=cloud2)
            hf.create_dataset('cloud3', data=cloud3)
            hf.close()

            if testing == True:
                break

    def voxel_grid_from_csv(self, src_path, labels_file, shape):
        src_suffix = ".h5"
        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        with open(labels_file, mode='r') as infile:
            reader = csv.reader(infile)
            labels = {rows[0]: rows[1] for rows in reader}
            infile.close()

        total_len = len(files)

        np.random.seed(100)
        randomized_index = np.arange(total_len)
        np.random.shuffle(randomized_index)

        layers = shape[0]
        depth = shape[1]
        height = shape[2]
        width = shape[3]

        X = np.zeros((total_len, layers, depth, height, width))
        y = np.zeros(total_len)

        sample_no = 0
        for f in tqdm(files):
            hf = h5py.File(f, 'r')
            cloud1 = hf['cloud1'][:]
            cloud2 = hf['cloud2'][:]
            cloud3 = hf['cloud3'][:]
            hf.close()

            sample_name = self.extract_name(f, src_path, src_suffix)

            label = labels[sample_name + ".mp4"]

            index = randomized_index[sample_no]
            X[index, 0, :, :, :] = ImageProcessor3D().voxel_grid_from_cloud(cloud1, out_depth=depth, out_height=height, out_width=width)
            X[index, 1, :, :, :] = ImageProcessor3D().voxel_grid_from_cloud(cloud2, out_depth=depth, out_height=height, out_width=width)
            X[index, 2, :, :, :] = ImageProcessor3D().voxel_grid_from_cloud(cloud3, out_depth=depth, out_height=height, out_width=width)
            y[index] = int(label)

            sample_no = sample_no + 1

        return X, y


def create_voxel_dataset_with_labels():

    src_path = "./data/point_cloud/"
    labels_file = "./data/labels.csv"

    grid_shape = [3, 32, 64, 64]
    X, y = Generator().voxel_grid_from_csv(src_path, labels_file, grid_shape)


if __name__ == "__main__":
    # Location of original dataset
    src_directory = "../../dataset/micro/"

    # Target Location of converted dataset
    dst_directory = "../../micro_extracted"

    # Generator().generate_processed_dataset(src_directory, dst_directory, testing=True)
    # Generator().generate_point_cloud_dataset(src_directory, dst_directory, testing=True)
    # create_voxel_dataset_with_labels()

    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

    print("Finished")
