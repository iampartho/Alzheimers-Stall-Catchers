import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from preprocess_images import VideoProcessor, ImageProcessor3D
import visualization_tools


class Tester:

    def __init__(self):
        self.data = []

    def test2_2(self):
        filename = "../../micro/100109.mp4"
        extractor = VideoProcessor(filename)
        extracted_images = extractor.process_video(roi_extraction=False, filter_enabled=False, average_frames=False)

        extracted_images_filtered = ImageProcessor3D().gaussian_filter_3D(extracted_images)
        extracted_images_filtered = ImageProcessor3D().create_time_chunks(extracted_images_filtered)
        extracted_images = ImageProcessor3D().create_time_chunks(extracted_images)

        # os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

        visualization_tools.CompareChunks(extracted_images, extracted_images_filtered)
        # visualization_tools.PlotIntensity(extracted_images)



if __name__ == "__main__":
    # instance = Tester()
    # instance.test2_1()
    # instance.test2_2()

    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
