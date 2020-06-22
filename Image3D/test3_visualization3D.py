import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from preprocess_images import VideoProcessor, ImageProcessor3D
import visualization_tools
from dataset_generator import Generator


class Tester:

    def __init__(self):
        self.data = []

    def test2_2(self):
        filename = "../../dataset/micro/115374.mp4"
        extractor = VideoProcessor(filename)
        extracted_images = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

        # visualization_tools.Interactive(extracted_images).compare_with_chunk(extracted_images_filtered)
        # visualization_tools.Interactive(extracted_images).plot_intensities()
        visualization_tools.Interactive(extracted_images).show_point_cloud(threshold=100, clustering=True, filter_outliers=True)

    def test2_3(self):
        checkpoint_enabled = True
        checkpoint = '117903'
        flag = False

        source_directory = "../../dataset/micro/"
        source_file_suffix = ".mp4"

        # Extract all processed frames from video files
        file_handler = Generator(source_directory, source_file_suffix)

        for f in file_handler.files:
            # Iterates through all the video files and export images
            plot_title = file_handler.extract_name(f)

            if checkpoint_enabled:
                if plot_title == checkpoint:
                    flag = True
                if flag == False:
                    continue

            extractor = VideoProcessor(f)
            extracted_images = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)
            visualization_tools.Interactive(extracted_images).show_point_cloud(threshold=100, clustering=True,
                                                                               filter_outliers=False, name=plot_title)


if __name__ == "__main__":
    instance = Tester()
    # instance.test2_2()
    instance.test2_3()

    # os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
