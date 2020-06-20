import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from matplotlib.widgets import Cursor, Button

from preprocess_images import VideoProcessor


class Visualizer3D:

    def __init__(self):
        self.data = []

    def tester4(self):
        '''
        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.subplot(2, 3, 2)
        plt.imshow(gamma_corrected, cmap='gray')
        plt.subplot(2, 3, 3)
        plt.imshow(blood_vessel_map, cmap='gray')

        plt.subplot(2, 3, 4)
        plt.hist(img.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        plt.subplot(2, 3, 5)
        plt.hist(gamma_corrected.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        plt.subplot(2, 3, 6)
        plt.hist(blood_vessel_map.ravel(), bins=256, range=(0, 255), fc='k', ec='k')

        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show()
        '''

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

    def convert_collection_to_grayscale(self, image_collection):
        no_frames = image_collection.shape[0]
        height = image_collection.shape[1]
        width = image_collection.shape[2]

        image_collection_grayscale = np.zeros((no_frames, height, width), dtype=np.uint8)
        for frame_no in range(no_frames):
            grayscaled = cv2.cvtColor(image_collection[frame_no, :, :, :], cv2.COLOR_BGR2GRAY)
            image_collection_grayscale[frame_no, :, :] = grayscaled

        return image_collection_grayscale

    def interactive_plot(self, image_collection):
        depth, height, width = image_collection.shape

        z_coord = math.floor(depth/2)
        y_coord = math.floor(height/2)
        x_coord = math.floor(width/2)

        blank_image = np.zeros_like(image_collection[0, :, :])

        fig = plt.figure()
        main_plot = plt.subplot(2, 2, 1)
        pixels_vertical = plt.subplot(2, 2, 2)
        pixels_horizontal = plt.subplot(2, 2, 3)
        pixels_depth = plt.subplot(2, 2, 4)

        main_plot.imshow(image_collection[z_coord, :, :], cmap='gray')
        main_plot.plot([x_coord, x_coord], [0, height-1], color='blue')
        main_plot.plot([0, width-1], [y_coord, y_coord], color='blue')

        pixels_horizontal.imshow(blank_image, cmap='gray')
        pixels_vertical.imshow(blank_image, cmap='gray')

        pixels_vertical.plot(image_collection[z_coord, :, x_coord], np.arange(0, height))
        pixels_horizontal.plot(np.arange(0, width), image_collection[z_coord, y_coord, :])
        pixels_depth.plot(np.arange(0, depth), image_collection[:, y_coord, x_coord])

        def on_click(event):
            x1, y1 = event.xdata, event.ydata
            print("Pixel ", x1, y1)

        def on_scroll(event):
            if event.button == 'up':
                print('up')
            elif event.button == 'down':
                print('down')

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        plt.show()


if __name__ == "__main__":

    filename = "../../micro/100109.mp4"
    extractor = VideoProcessor(filename)
    extracted_images = extractor.process_video(roi_extraction=False, average_frames=True)
    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

    visualizer = Visualizer3D()
    extracted_images = visualizer.convert_collection_to_grayscale(extracted_images)
    frame = extracted_images[0, :, :]
    # visualizer.surface3D(frame)
    # visualizer.point_cloud(frame)
    visualizer.interactive_plot(extracted_images)

