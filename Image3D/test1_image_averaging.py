import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ImageProcessor:
    no_countour = {"Filename": [], "Frame_number": []}

    def __init__(self):
        self.data = []

    def set_video_name(self, name):
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
        cv2.floodFill(im_floodfill, masked, (0, 0), 255);
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
            binaryFrame = ImageProcessor().segfromframe(frame)
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
        ImageProcessor().plotting3D(imx, imy, imz)

    def plotting3D(self, imx, imy, imz):

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(imx, imy, imz, 'gray')
        plt.show()

    def processFrame1(self, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 195, 205])
        upper_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        result = cv2.bitwise_and(img, img, mask=mask)

        im_floodfill = mask

        h, w = mask.shape[:2]
        masked = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(im_floodfill, masked, (0, 0), 255);

        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        im_out = img.copy()
        im_out[:, :, 0] = im_floodfill_inv
        im_out[:, :, 1] = im_floodfill_inv
        im_out[:, :, 2] = im_floodfill_inv
        # Combine the two images to get the foreground.
        # im_out = frame | im_floodfill_inv_img
        im_out = img & im_out

        binarized = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
        binarized = cv2.medianBlur(binarized, 5)
        binarized = cv2.adaptiveThreshold(binarized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

        return im_out, binarized

    def tester2(self):

        video = cv2.VideoCapture(self.video_name)
        z = 0
        while z < 5:
            ret, frame = video.read()
            if ret == False:
                break
            cropped, binaryFrame = ImageProcessor().processFrame1(frame)

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

    def tester2_1(self):

        video = cv2.VideoCapture(self.video_name)
        z = 0
        synthesizedImage = []
        use_images = 5
        plt_idx = 101 + 10*(use_images+1)
        while z < use_images:
            z = z + 1
            ret, frame = video.read()
            if ret == False:
                break
            #cropped, binaryFrame = ImageProcessor().processFrame1(frame)
            if z == 1:
                synthesizedImage = frame
            else:
                synthesizedImage = synthesizedImage + frame

            synthesizedImage[synthesizedImage<127] = 0

            plt.subplot(plt_idx)
            plt.imshow(frame)
            plt_idx = plt_idx + 1

        video.release()
        plt.subplot(plt_idx)
        plt.imshow(synthesizedImage)
        plt.show()

    def tester2_2(self):

        video = cv2.VideoCapture(self.video_name)
        z = 0
        image_accumulated = []
        cropped_image_accumulated = []
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_no in range(num_frames):
            ret, frame = video.read()
            if ret == False:
                break
            cropped, binaryFrame = ImageProcessor().processFrame1(frame)
            if frame_no == 0:
                image_accumulated = np.uint32(frame)
                cropped_image_accumulated = np.uint32(cropped)
                plt.subplot(131)
                plt.imshow(frame)
            else:
                image_accumulated = image_accumulated + np.uint32(frame)
                cropped_image_accumulated = cropped_image_accumulated + np.uint32(cropped)


        video.release()
        max_pixel_value = np.max(image_accumulated)
        #image_accumulated[image_accumulated < max_pixel_value / 5] = 0
        #cropped_image_accumulated[cropped_image_accumulated < max_pixel_value / 5] = 0
        image_accumulated = image_accumulated/max_pixel_value * 255
        cropped_image_accumulated = cropped_image_accumulated/max_pixel_value * 255
        image_accumulated = np.uint8(image_accumulated)
        cropped_image_accumulated = np.uint8(cropped_image_accumulated)
        plt.subplot(132)
        plt.imshow(image_accumulated)
        plt.subplot(133)
        plt.imshow(cropped_image_accumulated)
        plt.show()

    def tester2_3(self):

        video = cv2.VideoCapture(self.video_name)
        z = 0

        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        number_of_chunks = math.floor(num_frames/10)
        #print(num_frames, number_of_chunks)
        for iterations in range(number_of_chunks):
            image_accumulated = []
            cropped_image_accumulated = []
            for frame_no in range(10):
                ret, frame = video.read()
                if ret == False:
                    break
                cropped, binaryFrame = ImageProcessor().processFrame1(frame)
                if frame_no == 0:
                    image_accumulated = np.uint32(frame)
                    cropped_image_accumulated = np.uint32(cropped)
                    plt.subplot(131)
                    plt.imshow(frame)
                else:
                    image_accumulated = image_accumulated + np.uint32(frame)
                    cropped_image_accumulated = cropped_image_accumulated + np.uint32(cropped)

            max_pixel_value = np.max(image_accumulated)
            image_accumulated[image_accumulated < max_pixel_value / 3] = 0
            cropped_image_accumulated[cropped_image_accumulated < max_pixel_value / 3] = 0
            image_accumulated = image_accumulated/max_pixel_value * 255
            cropped_image_accumulated = cropped_image_accumulated/max_pixel_value * 255
            image_accumulated = np.uint8(image_accumulated)
            cropped_image_accumulated = np.uint8(cropped_image_accumulated)
            plt.subplot(132)
            plt.imshow(image_accumulated)
            plt.subplot(133)
            plt.imshow(cropped_image_accumulated)
            plt.show()

        video.release()

    def tester2_4(self):

        video = cv2.VideoCapture(self.video_name)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(3))
        height = int(video.get(4))
        layers = 3

        image_collection = np.zeros((num_frames, height, width, layers))
        cropped_image_collection = np.zeros((num_frames, height, width, layers))

        num_frame_overlap = 5
        num_frame_chunk = 10
        num_chunks = 1 + math.floor((num_frames-num_frame_chunk)/num_frame_overlap)

        image_collection_averaged = np.zeros((num_chunks, height, width, layers))
        cropped_image_collection_averaged = np.zeros((num_chunks, height, width, layers))

        for frame_no in range(num_frames-1):

            ret, frame = video.read()
            if ret == False:
                break

            cropped, binaryFrame = ImageProcessor().processFrame1(frame)
            image_collection[frame_no, :, :, :] = frame
            cropped_image_collection[frame_no, :, :, :] = cropped

        for chunk_no in range(num_chunks):

            start_frame = chunk_no * num_frame_overlap

            augmented_frame = np.zeros((height, width, layers))
            augmented_cropped_frame = np.zeros((height, width, layers))

            for frame_no in range(start_frame, start_frame + num_frame_chunk):
                augmented_frame = augmented_frame + np.uint32(image_collection[frame_no, :, :, :])
                augmented_cropped_frame = augmented_cropped_frame + np.uint32(cropped_image_collection[frame_no, :, :, :])

            max_pixel_value = np.max(augmented_frame)
            # augmented_frame[augmented_frame < max_pixel_value / 3] = 0
            # augmented_cropped_frame[augmented_cropped_frame < max_pixel_value / 3] = 0
            augmented_frame = augmented_frame / max_pixel_value * 255
            augmented_cropped_frame = augmented_cropped_frame / max_pixel_value * 255
            augmented_frame = np.uint8(augmented_frame)
            augmented_cropped_frame = np.uint8(augmented_cropped_frame)

            image_collection_averaged[chunk_no, :, :, :] = augmented_frame
            cropped_image_collection_averaged[chunk_no, :, :, :] = augmented_cropped_frame

            plt.subplot(121)
            plt.imshow(augmented_frame)
            plt.subplot(122)
            plt.imshow(augmented_cropped_frame)
            plt.show()

        video.release()
        print(image_collection.shape)


testObject = ImageProcessor()
testObject.set_video_name('../../micro/100289.mp4')
testObject.tester2_4()
