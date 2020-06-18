import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:
    no_countour = {"Filename": [], "Frame_number": []}

    def __init__(self, name):
        self.data = []
        self.video_name = name

    def segfromframe(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 195, 205])
        upper_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        im_floodfill = mask

        h, w = mask.shape[:2]
        masked = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, masked, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        im_out = frame.copy()
        im_out[:, :, 0] = im_floodfill_inv
        im_out[:, :, 1] = im_floodfill_inv
        im_out[:, :, 2] = im_floodfill_inv
        # Combine the two images to get the foreground.
        # im_out = frame | im_floodfill_inv_img
        im_out = frame & im_out

        # finding contour
        cnts = cv2.findContours(im_floodfill_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Find bounding box and extract ROI
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # ROI = frame[y-3:y+h+3, x-3:x+w+3]
            ROI = im_out[y:y + h, x:x + w]
            ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            ROI = cv2.medianBlur(ROI, 5)
            ret, ROI = cv2.threshold(ROI, 127, 255, cv2.THRESH_BINARY)
            break

        try:
            return ROI
        except:
            return frame

    def tester(self):

        cap = cv2.VideoCapture('test.mp4')
        z = 0

        imx = []
        imy = []
        imz = []

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            binaryFrame = self.segfromframe(frame)
            # plt.imshow(binaryFrame, 'gray')
            # plt.show()

            height, width = binaryFrame.shape
            for x in range(height):
                for y in range(width):
                    if binaryFrame[x, y]:
                        imx.append(x)
                        imy.append(y)
                        imz.append(z)

            z += 0.1

        imx = np.array(imx)
        imy = np.array(imy)
        imz = np.array(imz)

        cap.release()
        self.plotting3D(imx, imy, imz)

    def plotting3D(self, imx, imy, imz):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(imx, imy, imz, 'gray')
        plt.show()

    def process_frame_1(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 195, 205])
        upper_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        im_floodfill = mask

        h, w = mask.shape[:2]
        masked = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(im_floodfill, masked, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        im_out = img.copy()
        im_out[:, :, 0] = im_floodfill_inv
        im_out[:, :, 1] = im_floodfill_inv
        im_out[:, :, 2] = im_floodfill_inv
        im_out = img & im_out

        binarized = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
        binarized = cv2.medianBlur(binarized, 5)
        binarized = cv2.adaptiveThreshold(binarized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return im_out, binarized

    def tester2(self):
        video = cv2.VideoCapture(self.video_name)
        z = 0
        while z < 5:
            ret, frame = video.read()
            if ret == False:
                break
            cropped, binaryFrame = self.process_frame_1(frame)

            plt.subplot(131)
            plt.imshow(frame)
            plt.subplot(132)
            plt.imshow(cropped)
            plt.subplot(133)
            plt.imshow(binaryFrame)
            # plt.imshow(binaryFrame, 'gray')
            plt.show()

            z = z + 1

        video.release()

    def compand(self, img, mu):
        img = 2 * img.astype(float) / 255.0 - 1
        companded = np.multiply(np.sign(img), np.log(1+mu*np.abs(img))) / np.log(1+mu)
        companded = (companded + 1)/2*255
        return np.uint8(companded)

    def process_frame_2(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(img, cv2.CV_8U)
        # blurred = cv2.GaussianBlur(img, (5, 5), 0)
        # blurred = cv2.bilateralFilter(img, 9, 75, 75)
        # companded = self.compand(blurred, 255)

        return laplacian

    def normalize_wrt_percentile(self, img, percent_val):
        overbright_pixel_val = np.percentile(img, percent_val)
        img[img > overbright_pixel_val] = overbright_pixel_val
        return np.uint8(img / overbright_pixel_val * 255)


    def extract_video_frames(self, video):
        no_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(3))
        height = int(video.get(4))
        layers = 3

        image_collection = np.zeros((no_frames, height, width, layers), dtype=np.uint8)
        for frame_no in range(no_frames):
            ret, frame = video.read()
            if ret == False:
                break
            image_collection[frame_no, :, :, :] = frame

        return image_collection

    def create_vessel_map(self, image_collection):
        no_frames = image_collection.shape[0]
        height = image_collection.shape[1]
        width = image_collection.shape[2]
        layers = 3

        vessel_map = np.zeros((height, width, layers), dtype=np.uint32)
        for frame_no in range(no_frames):
            vessel_map = vessel_map + np.uint32(image_collection[frame_no, :, :, :])

        return np.uint8(vessel_map/no_frames)

    def create_time_chunks(self, image_collection, frame_overlap, chunk_size):
        no_frames = image_collection.shape[0]
        height = image_collection.shape[1]
        width = image_collection.shape[2]
        layers = 3
        num_chunks = 1 + math.floor((no_frames - chunk_size) / frame_overlap)

        image_collection_averaged = np.zeros((num_chunks, height, width, layers), dtype=np.uint8)

        for chunk_no in range(num_chunks):
            start_frame = chunk_no * frame_overlap
            averaged_frame = np.zeros((height, width, layers), dtype=np.uint8)
            for frame_no in range(start_frame, start_frame + chunk_size):
                averaged_frame = averaged_frame + np.uint32(image_collection[frame_no, :, :, :])

            averaged_frame = np.uint8(averaged_frame/chunk_size)
            image_collection_averaged[chunk_no, :, :, :] = averaged_frame

        return image_collection_averaged

    def tester2_6(self):
        video = cv2.VideoCapture(self.video_name)
        image_collection = self.extract_video_frames(video)
        video.release()

        blood_vessel_map = self.create_vessel_map(image_collection)
        blood_vessel_map = cv2.cvtColor(blood_vessel_map, cv2.COLOR_BGR2GRAY)
        image_collection_averaged = self.create_time_chunks(image_collection, frame_overlap=5, chunk_size=10)

        for frame in range(image_collection_averaged.shape[0]):
            image = image_collection_averaged[frame, :, :, :]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.subplot(2, 2, 1)
            plt.imshow(blood_vessel_map, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 2, 3)
            plt.hist(blood_vessel_map.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
            plt.subplot(2, 2, 4)
            plt.hist(image.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
            plt.show()


testObject = ImageProcessor('../../micro/105159.mp4')
testObject.tester2_6()
