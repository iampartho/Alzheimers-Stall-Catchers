import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess_images import VideoProcessor, ImageProcessor3D


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


class Interactive:

    def __init__(self, image_collection):
        self.image_collection = image_collection

        self.depth, self.height, self.width = self.image_collection.shape
        self.z = math.floor(self.depth / 2)
        self.y = math.floor(self.height / 2)
        self.x = math.floor(self.width / 2)

        self.fig = plt.figure()

    def callback_click(self, event, update_function):
        self.x, self.y = math.floor(event.xdata), math.floor(event.ydata)
        update_function()

    def callback_scroll(self, event, update_function):
        if event.button == 'up':
            self.z = self.z + 1
        elif event.button == 'down':
            self.z = self.z - 1
        update_function()

    def compare_with_chunk(self, processed_collection):
        self.processed_collection = processed_collection
        self.main_img = plt.subplot(1, 2, 1)
        self.processed_img = plt.subplot(1, 2, 2)

        def update_figure():
            if self.z >= self.depth:
                self.z = self.depth - 1
            elif self.z < 0:
                self.z = 0

            self.main_img.cla()
            self.processed_img.cla()

            self.main_img.imshow(self.image_collection[self.z, :, :], cmap='gray')
            self.main_img.title.set_text('Frame no ' + str(self.z))

            self.processed_img.imshow(self.processed_collection[self.z, :, :], cmap='gray')
            self.processed_img.title.set_text(' Processed Frame ' + str(self.z))

            plt.draw()

        self.fig.canvas.mpl_connect('scroll_event', lambda event: self.callback_scroll(event, update_figure))
        update_figure()
        plt.show()

    def plot_intensities(self):
        self.blank_frame = np.zeros_like(self.image_collection[0, :, :])
        self.main_plot = plt.subplot(2, 2, 1)
        self.pixels_intensity_vertical = plt.subplot(2, 2, 2)
        self.pixels_intensity_horizontal = plt.subplot(2, 2, 3)
        self.pixels_intensity_depth = plt.subplot(2, 2, 4)

        def update_plots():
            if self.z >= self.depth:
                self.z = self.depth - 1
            elif self.z < 0:
                self.z = 0

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

        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.callback_click(event, update_plots))
        self.fig.canvas.mpl_connect('scroll_event', lambda event: self.callback_scroll(event, update_plots))
        update_plots()
        plt.show()

    def show_point_cloud(self, percentile=95, clustering=False, filter_outliers=False, name=''):
        self.cloud_plot = self.fig.add_subplot(111, projection='3d')
        self.z = percentile

        def update_cloud():
            if self.z >= 100:
                self.z = 100
            elif self.z < 0:
                self.z = 0

            self.cloud_plot.cla()

            # Generate point cloud
            thresh = int(np.percentile(self.image_collection.ravel(), self.z))
            cloud, labels = ImageProcessor3D().point_cloud_from_collecton(self.image_collection, threshold=thresh, filter_outliers=filter_outliers)

            plot_title = 'Video ' + name + ' Percentile:' + str(self.z) + ' Threshold: ' + str(thresh)
            self.cloud_plot.title.set_text(plot_title)

            if clustering:
                self.cloud_plot.scatter(cloud[:, 2], cloud[:, 0], cloud[:, 1], marker='.', c=labels, cmap='viridis')
            else:
                self.cloud_plot.scatter(cloud[:, 2], cloud[:, 0], cloud[:, 1], marker='.', color='#990000')

            self.cloud_plot.set_xlabel('Width')
            self.cloud_plot.set_ylabel('Depth')
            self.cloud_plot.set_zlabel('Height')

            limit = max(self.height, self.width, self.depth)
            self.cloud_plot.set_xlim([0, limit])
            self.cloud_plot.set_ylim([0, limit])
            self.cloud_plot.set_zlim([0, limit])

            plt.gca().invert_zaxis()
            elevation = 30  # Up/Down
            azimuth = 300  # Left/Right
            self.cloud_plot.view_init(elevation, azimuth)

            plt.draw()

        self.fig.canvas.mpl_connect('scroll_event', lambda event: self.callback_scroll(event, update_cloud))
        update_cloud()
        plt.show()


if __name__ == "__main__":

    filename = "../../micro/100109.mp4"
    extractor = VideoProcessor(filename)
    extracted_images = extractor.process_video(roi_extraction=False, average_frames=True)
    # os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

    Interactive(extracted_images).plot_intensities()
