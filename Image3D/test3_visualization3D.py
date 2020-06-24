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
        filename = "../../dataset/micro/100109.mp4"
        extractor = VideoProcessor(filename)
        image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

        # visualization_tools.Interactive(extracted_images).compare_with_chunk(extracted_images_filtered)
        # visualization_tools.Interactive(extracted_images).plot_intensities()
        # plt.hist(extracted_images.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        # plt.show()
        visualization_tools.Interactive(image_collection).show_point_cloud(percentile=95, clustering=True, filter_outliers=True)

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
    instance.test2_2()
    # instance.test2_3()
    # instance.test2_3_1()

    # os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
