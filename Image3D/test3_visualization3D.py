import math
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import csv
from tqdm import tqdm

from preprocess_images import VideoProcessor, ImageProcessor3D
import visualization_tools
from dataset_generator import Generator


class Tester:

    def __init__(self):
        self.data = []

    def test2_4_voxel_hf(self):
        filename = "../../dataset/micro/100109.mp4"
        extractor = VideoProcessor(filename)
        image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

        print(image_collection.shape)
        layer = 3
        depth = image_collection.shape[0]
        width = image_collection.shape[1]
        height = image_collection.shape[2]
        voxel_grid = np.zeros((layer, depth, width, height), np.uint8)

        thresh0 = int(np.percentile(image_collection.ravel(), 99))
        thresh1 = int(np.percentile(image_collection.ravel(), 95))
        thresh2 = int(np.percentile(image_collection.ravel(), 90))

        voxel_grid[0, :, :, :] = ImageProcessor3D().grayscale_collection_to_binary(image_collection, threshold=thresh0)
        voxel_grid[1, :, :, :] = ImageProcessor3D().grayscale_collection_to_binary(image_collection, threshold=thresh1)
        voxel_grid[2, :, :, :] = ImageProcessor3D().grayscale_collection_to_binary(image_collection, threshold=thresh2)

        hf = h5py.File('data.h5', 'w')
        hf.create_dataset('voxel', data=voxel_grid)
        hf.close()

        h5f = h5py.File('data.h5', 'r')
        b = h5f['voxel'][:]
        h5f.close()

        print(b.shape)

    def test2_5_voxel_hf_multi(self):
        source_directory = "../../dataset/micro/"
        source_file_suffix = ".mp4"

        labels_file = "../../dataset/csv_files/train_labels.csv"

        file_handler = Generator(source_directory, source_file_suffix)

        with open(labels_file, mode='r') as infile:
            reader = csv.reader(infile)
            labels = {rows[0]: rows[1] for rows in reader}

        train_size = 20
        test_size = 5

        nominal_layers = 3
        nominal_depth = 64
        nominal_height = 128
        nominal_width = 128

        X_train = np.zeros((train_size, nominal_layers, nominal_depth, nominal_height, nominal_width), dtype=np.uint8)
        y_train = np.zeros(train_size, dtype=np.uint8)
        X_test = np.zeros((test_size, nominal_layers, nominal_depth, nominal_height, nominal_width), dtype=np.uint8)
        y_test = np.zeros(test_size, dtype=np.uint8)

        data_no = 0
        for f in file_handler.files:
            # Iterates through all the video files and export images
            sample_name = file_handler.extract_name(f)
            label = labels[sample_name + '.mp4']

            extractor = VideoProcessor(f)
            image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

            layer = 3
            depth = image_collection.shape[0]
            width = image_collection.shape[1]
            height = image_collection.shape[2]
            voxel_grid = np.zeros((layer, depth, width, height), np.uint8)

            thresh0 = int(np.percentile(image_collection.ravel(), 99))
            thresh1 = int(np.percentile(image_collection.ravel(), 95))
            thresh2 = int(np.percentile(image_collection.ravel(), 90))

            voxel_grid[0, :, :, :] = ImageProcessor3D().grayscale_collection_to_binary(image_collection,
                                                                                       threshold=thresh0)
            voxel_grid[1, :, :, :] = ImageProcessor3D().grayscale_collection_to_binary(image_collection,
                                                                                       threshold=thresh1)
            voxel_grid[2, :, :, :] = ImageProcessor3D().grayscale_collection_to_binary(image_collection,
                                                                                       threshold=thresh2)

            if depth > nominal_depth:
                depth = nominal_depth
                voxel_grid = voxel_grid[:, :nominal_depth, :, :]
            if width > nominal_width:
                width = nominal_width
                voxel_grid = voxel_grid[:, :, :nominal_width, :]
            if height > nominal_height:
                height = nominal_height
                voxel_grid = voxel_grid[:, :, :, :nominal_height]

            depth_start = math.floor((nominal_depth - depth)/2)
            width_start = math.floor((nominal_width - width)/2)
            height_start = math.floor((nominal_height -height)/2)

            if (data_no < train_size):
                X_train[data_no, :, depth_start:depth_start+depth, width_start:width_start+width,
                            height_start:height_start+height] = voxel_grid
                y_train[data_no] = int(label)
            else:
                X_test[data_no-train_size, :, depth_start:depth_start+depth, width_start:width_start+width,
                            height_start:height_start+height] = voxel_grid
                y_test[data_no-train_size] = int(label)

            data_no = data_no + 1
            if data_no == train_size+test_size:
                break

        hf = h5py.File('../../data.h5', 'w')
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('X_test', data=X_test)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('y_test', data=y_test)
        hf.close()

    def test2_5_1(self):
        h5f = h5py.File('../../data.h5', 'r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        h5f.close()

    def test2_3(self):
        checkpoint_enabled = True
        checkpoint = '146244'
        flag = False

        source_directory = "../../dataset/micro/"
        source_file_suffix = ".mp4"

        # Extract all processed frames from video files
        file_handler = Generator(source_directory, source_file_suffix)

        for f in tqdm(file_handler.files):
            # Iterates through all the video files and export images
            plot_title = file_handler.extract_name(f)

            if checkpoint_enabled:
                if plot_title == checkpoint:
                    flag = True
                if flag == False:
                    continue

            extractor = VideoProcessor(f)
            extracted_images = extractor.process_video(roi_extraction=True, filter_enabled=False, average_frames=True)

            percentiles = [99, 95, 90]
            for i in range(3):
                visualization_tools.Interactive(extracted_images).show_point_cloud(percentile=percentiles[i], clustering=True,
                                                                               filter_outliers=True, name=plot_title)


    def test2_3_1(self):
        source_directory = "../../dataset/micro/"
        source_file_suffix = ".mp4"

        file_handler = Generator(source_directory, source_file_suffix)

        depths = np.zeros(2399)
        heights = np.zeros(2399)
        widths = np.zeros(2399)
        i = 0

        for f in tqdm(file_handler.files):
            extractor = VideoProcessor(f)
            extracted_images = extractor.process_video(roi_extraction=True, filter_enabled=False, average_frames=True)

            depths[i] = extracted_images.shape[0]
            heights[i] = extracted_images.shape[1]
            widths[i] = extracted_images.shape[2]

            i = i + 1

        print("Depth min: " + str(np.min(depths)) + " average: " + str(np.average(depths)) + " max: " + str(
            np.max(depths)))
        print(np.percentile(depths, 90), np.percentile(depths, 95), np.percentile(depths, 99))
        print("Height min: " + str(np.min(heights)) + " average: " + str(np.average(heights)) + " max: " + str(
            np.max(heights)))
        print(np.percentile(heights, 90), np.percentile(heights, 95), np.percentile(heights, 99))
        print("Width min: " + str(np.min(widths)) + " average: " + str(np.average(widths)) + " max: " + str(
            np.max(widths)))
        print(np.percentile(widths, 90), np.percentile(widths, 95), np.percentile(widths, 99))

    def test2_2(self):
        filename = "../../dataset/micro/102750.mp4"
        extractor = VideoProcessor(filename)
        image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

        # visualization_tools.Interactive(extracted_images).compare_with_chunk(extracted_images_filtered)
        # visualization_tools.Interactive(extracted_images).plot_intensities()
        # plt.hist(extracted_images.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        # plt.show()
        visualization_tools.Interactive(image_collection).show_point_cloud(percentile=95, clustering=False, filter_outliers=True)

    def test2_2_1(self):
        filename = "../../dataset/micro/187558.mp4"
        extractor = VideoProcessor(filename)
        image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

        thresh = int(np.percentile(image_collection.ravel(), 95))
        cloud, labels = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh,
                                                                      filter_outliers=True)
        # voxels = ImageProcessor3D().voxel_grid_from_cloud(cloud, out_depth=50, out_width=128, out_height=128)
        # cloud2 = np.argwhere(voxels == 255)

        h5f = h5py.File('../../point_cloud_dataset/187558.h5', 'r')
        cloud2 = h5f['cloud2'][:]
        h5f.close()

        print(cloud2.shape[0], cloud.shape[0])


if __name__ == "__main__":
    instance = Tester()
    instance.test2_2_1()
    # instance.test2_3()
    # instance.test2_3_1()
    # instance.test2_4_voxel_hf()
    # instance.test2_5_voxel_hf_multi()
    # instance.test2_5_1()

    # os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
