import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

from preprocess_images import VideoProcessor


class Tool3D:

    def __init__(self):
        self.data = []

    def show_histogram(self, img):
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.hist(img.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show()

    def surface3D(self, img):
        height, width = img.shape
        x = np.outer(np.arange(0, width), np.ones(height)).T
        y = np.outer(np.flip(np.arange(0, height)), np.ones(width))
        z = cv2.GaussianBlur(img, (5, 5), 1)
        z = z.astype(float)/255.0
        z = 3*np.power(z, 2) - 2*np.power(z, 3)

        img = img.astype(float)/255.0
        img = 3*np.power(img, 2) - 2*np.power(img, 3)

        fig = plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122, projection='3d')

        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original image')
        ax2.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
        ax2.set_title('Surface plot')
        ax2.set_xlabel('x axis')
        ax2.set_ylabel('y axis')
        plt.show()

    def point_cloud(self, depth_map):

        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 1)

        start_time = time()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        rows, cols = depth_map.shape

        count = 0

        # Cut down pixels for time purpose
        # Computations need to be under 30s
        pixel_cut = 3

        # Iterate thorugh all the pixels
        X = []
        Y = []
        Z = []
        for x in range(cols):
            for y in range(rows):
                if (x % pixel_cut == 0 and y % pixel_cut == 0):
                    if depth_map[y,x] > 100:
                        count += 1
                        depth = depth_map[y, x]
                        depth = (float(depth) - 100.0) / (255.0 - 100.0)
                        depth = 3*math.pow(depth, 2) - 2*math.pow(depth, 3)

                        X.append(x)
                        Y.append(y)
                        Z.append(depth)

        print('Finished loop, tryna plot')
        # Axis Labels
        ax.scatter(X, Z, Y, marker='*')
        ax.set_xlabel('Width')
        ax.set_ylabel('Depth')
        ax.set_zlabel('Height')

        plt.gca().invert_zaxis()

        ###########################################
        # Play with me to change view rotation!
        elevation = 30  # Up/Down
        azimuth = 300  # Left/Right
        ###########################################

        ax.view_init(elevation, azimuth)

        plt.show()  # Uncomment if running on your local machine
        print("Outputted {} of the {} points".format(count, 6552))
        print("Results produced in {:04.2f} seconds".format(time() - start_time))


class PlotIntensity:

    def __init__(self, image_collection):
        self.image_collection = image_collection

        self.depth, self.height, self.width = self.image_collection.shape

        self.z = math.floor(self.depth / 2)
        self.y = math.floor(self.height / 2)
        self.x = math.floor(self.width / 2)

        self.blank_frame = np.zeros_like(self.image_collection[0, :, :])

        self.fig = plt.figure()
        self.main_plot = plt.subplot(2, 2, 1)
        self.pixels_intensity_vertical = plt.subplot(2, 2, 2)
        self.pixels_intensity_horizontal = plt.subplot(2, 2, 3)
        self.pixels_intensity_depth = plt.subplot(2, 2, 4)

        self.show()

    def show(self):

        def update_plots():
            self.main_plot.cla()
            self.pixels_intensity_vertical.cla()
            self.pixels_intensity_horizontal.cla()
            self.pixels_intensity_depth.cla()

            self.main_plot.imshow(self.image_collection[self.z, :, :], cmap='gray')
            self.main_plot.plot([self.x, self.x], [0, self.height-1], color='blue')
            self.main_plot.plot([0, self.width-1], [self.y, self.y], color='blue')
            self.main_plot.title.set_text('Frame no ' + str(self.z))

            self.pixels_intensity_vertical.imshow(self.blank_frame, cmap='gray')
            self.pixels_intensity_horizontal.imshow(self.blank_frame, cmap='gray')

            self.pixels_intensity_vertical.plot(self.image_collection[self.z, :, self.x], np.arange(0, self.height))
            self.pixels_intensity_horizontal.plot(np.arange(0, self.width), self.image_collection[self.z, self.y, :])
            self.pixels_intensity_depth.plot(np.arange(0, self.depth), self.image_collection[:, self.y, self.x])

            plt.draw()

        def on_click(event):
            self.x, self.y = math.floor(event.xdata), math.floor(event.ydata)
            print(self.x, self.y)
            update_plots()

        def on_scroll(event):
            if event.button == 'up':
                if self.z < self.depth - 1:
                    self.z = self.z + 1
            elif event.button == 'down':
                if self.z > 0:
                    self.z = self.z - 1
            print(self.z)
            update_plots()

        self.fig.canvas.mpl_connect('button_press_event', on_click)
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        update_plots()
        plt.show()


class CompareChunks:

    def __init__(self, image_collection, processed_collection):
        self.image_collection = image_collection
        self.processed_collection = processed_collection

        self.depth, self.height, self.width = self.image_collection.shape
        self.z = math.floor(self.depth / 2)

        self.fig = plt.figure()
        self.main_img = plt.subplot(1, 2, 1)
        self.processed_img = plt.subplot(1, 2, 2)

        self.show()

    def show(self):

        def update_plots():
            self.main_img.cla()
            self.processed_img.cla()

            self.main_img.imshow(self.image_collection[self.z, :, :], cmap='gray')
            self.main_img.title.set_text('Frame no ' + str(self.z))

            self.processed_img.imshow(self.processed_collection[self.z, :, :], cmap='gray')
            self.processed_img.title.set_text(' Processed Frame ' + str(self.z))

            plt.draw()

        def on_scroll(event):
            if event.button == 'up':
                if self.z < self.depth - 1:
                    self.z = self.z + 1
            elif event.button == 'down':
                if self.z > 0:
                    self.z = self.z - 1
            print(self.z)
            update_plots()

        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        update_plots()
        plt.show()


if __name__ == "__main__":

    filename = "../../micro/100109.mp4"
    extractor = VideoProcessor(filename)
    extracted_images = extractor.process_video(roi_extraction=False, average_frames=True)
    # os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

    PlotIntensity(extracted_images)
