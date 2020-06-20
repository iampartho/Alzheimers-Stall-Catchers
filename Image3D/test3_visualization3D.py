import os
import matplotlib.pyplot as plt
import numpy as np

from preprocess_images import VideoProcessor
from visualization_tools import Visualizer3D, InteractivePlotter
from filter_images import VoxelFilter

class Tester:

    def __init__(self):
        self.data = []

    def test1(self):
        filename = "../../micro/100109.mp4"
        extractor = VideoProcessor(filename)
        extracted_images = extractor.process_video(roi_extraction=False, average_frames=True)
        os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

        visualizer = Visualizer3D()
        extracted_images = visualizer.convert_collection_to_grayscale(extracted_images)
        frame = extracted_images[0, :, :]
        # visualizer.surface3D(frame)
        # visualizer.point_cloud(frame)
        # analyzer = InteractivePlotter(extracted_images)
        # analyzer.show()
        # plt.imshow(extracted_images[:, 200, :], cmap='gray')
        # plt.show()

    def test2_1(self):
        filename = "../../micro/100109.mp4"
        extractor = VideoProcessor(filename)
        extracted_images = extractor.process_video(roi_extraction=False, average_frames=False)
        os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

        visualizer = Visualizer3D()
        extracted_images = visualizer.convert_collection_to_grayscale(extracted_images)
        frame = extracted_images[0, :, :]

        filter_test = VoxelFilter()
        image_trimmed = frame.copy()
        height, width = frame.shape
        image_trimmed = image_trimmed[:height, :height]
        image_trimmed = (image_trimmed.astype(float)/255.0)*2 - 1.0
        image_filtered = np.zeros_like(image_trimmed)

        testing = False
        row_wise = False

        if testing:
            column = image_trimmed[:, 200]
            row = image_trimmed[200, :]
            column_filtered = filter_test.low_pass_filter(column)
            row_filtered = filter_test.low_pass_filter(row)

            plt.subplot(2, 2, 1)
            plt.plot(column)
            plt.subplot(2, 2, 2)
            plt.plot(row)
            plt.subplot(2, 2, 3)
            plt.plot(column_filtered)
            plt.subplot(2, 2, 4)
            plt.plot(row_filtered)
            plt.show()

        else:
            if row_wise:
                for pixel in range(height):
                    row = image_trimmed[pixel, :]
                    row_filtered = filter_test.low_pass_filter(row)
                    image_filtered[pixel, :] = row_filtered

                min = np.min(image_filtered)
                max = np.max(image_filtered)
                span = max - min
                image_filtered = np.uint8((image_filtered + min)/span * 255.0)
                res = np.hstack((frame, image_filtered))
                plt.imshow(res, cmap='gray')
                plt.show()

            else:
                for pixel in range(height):
                    column = image_trimmed[:, pixel]
                    column_filtered = filter_test.low_pass_filter(column)
                    image_filtered[:, pixel] = column_filtered

                min = np.min(image_filtered)
                max = np.max(image_filtered)
                span = max - min
                image_filtered = np.uint8((image_filtered + min) / span * 255.0)
                res = np.hstack((frame, image_filtered))
                plt.imshow(res, cmap='gray')
                plt.show()

if __name__ == "__main__":
    instance = Tester()
    # instance.test1()
    instance.test2_1()




