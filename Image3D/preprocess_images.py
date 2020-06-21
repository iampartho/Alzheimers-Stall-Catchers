import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class ImageProcessor:

    def __init__(self):
        self.data = []

    def compand(self, img, mu):
        img = 2 * img.astype(float) / 255.0 - 1
        companded = np.multiply(np.sign(img), np.log(1 + mu * np.abs(img))) / np.log(1 + mu)
        companded = (companded + 1) / 2 * 255
        return np.uint8(companded)

    def normalize_wrt_percentile(self, img, percent_val):
        overbright_pixel_val = np.percentile(img, percent_val)
        img[img > overbright_pixel_val] = overbright_pixel_val
        return np.uint8(img / overbright_pixel_val * 255)

    def equalize_histogram(self, img):
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
        equ = cv2.equalizeHist(img)
        res = np.hstack((img, equ))  # stacking images side-by-side
        return res

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)


class ImageProcessor3D:

    def __init__(self):
        self.data = []

    def gaussian_filter_3D(self, image_collection, input_sigma=1, filter_order=0):
        return gaussian_filter(image_collection, sigma=input_sigma, order=filter_order)

    def create_time_chunks(self, image_collection_grayscale, frame_overlap=5, chunk_size=10):
        # Take all the frames from video and export some averaged frames of given chunk size
        no_frames = image_collection_grayscale.shape[0]
        height = image_collection_grayscale.shape[1]
        width = image_collection_grayscale.shape[2]
        num_chunks = 1 + math.floor((no_frames - chunk_size) / frame_overlap)

        image_collection_averaged = np.zeros((num_chunks, height, width), dtype=np.uint8)

        for chunk_no in range(num_chunks):
            start_frame = chunk_no * frame_overlap
            averaged_frame = np.zeros((height, width), dtype=np.uint8)
            for frame_no in range(start_frame, start_frame + chunk_size):
                averaged_frame = averaged_frame + np.uint32(image_collection_grayscale[frame_no, :, :])

            averaged_frame = np.uint8(averaged_frame/chunk_size)
            image_collection_averaged[chunk_no, :, :] = averaged_frame

        return image_collection_averaged


class VideoProcessor:

    def __init__(self, name):
        self.data = []
        self.video_name = name

    def extract_ROI_from_collection(self, image_collection):
        no_frames = image_collection.shape[0]
        frame = image_collection[0, :, :, :]
        mask, boundingbox = self.find_ROI(frame)

        ROI_collection = np.zeros((no_frames, boundingbox[3], boundingbox[2], 3), dtype=np.uint8)

        for idx in range(no_frames):
            img = image_collection[idx, :, :, :]
            ROI_collection[idx, :, :, :] = self.extract_bounded_region(img, mask, boundingbox)

        return ROI_collection

    def find_ROI(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Threshold of blue in HSV space
        lower_orange = np.array([5, 195, 205])
        upper_orange = np.array([20, 255, 255])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.dilate(mask, kernel, iterations=1)

        im_floodfill = mask
        h, w = mask.shape[:2]
        masked = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, masked, (0, 0), 255);
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        cnts = cv2.findContours(im_floodfill_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        x, y, w, h = cv2.boundingRect(cnts[0])
        boundingbox = np.zeros(4).astype(int)
        boundingbox[0] = x
        boundingbox[1] = y
        boundingbox[2] = w
        boundingbox[3] = h

        return im_floodfill_inv, boundingbox

    def extract_bounded_region(self, frame, im_floodfill_inv, boundingbox):
        im_out = frame.copy()
        im_out[:, :, 0] = frame[:, :, 0] & im_floodfill_inv
        im_out[:, :, 1] = frame[:, :, 1] & im_floodfill_inv
        im_out[:, :, 2] = frame[:, :, 2] & im_floodfill_inv

        x = boundingbox[0]
        y = boundingbox[1]
        w = boundingbox[2]
        h = boundingbox[3]
        ROI = im_out[y:y + h, x:x + w, :]

        return ROI

    def extract_video_frames(self, video):
        # Extracts all frames from video to np array
        no_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(3))
        height = int(video.get(4))
        layers = 3

        image_collection = np.zeros((no_frames, height, width, layers), dtype=np.uint8)
        for frame_no in range(no_frames):
            ret, frame = video.read()
            if ret == False:
                break
            # frame = self.process_frame_1(frame)
            image_collection[frame_no, :, :, :] = frame

        return image_collection

    def create_vessel_map(self, image_collection_grayscale):
        # An average of all the frames of video sample creating an average map of all vessels
        no_frames = image_collection_grayscale.shape[0]
        height = image_collection_grayscale.shape[1]
        width = image_collection_grayscale.shape[2]

        vessel_map = np.zeros((height, width), dtype=np.uint32)
        for frame_no in range(no_frames):
            vessel_map = vessel_map + np.uint32(image_collection_grayscale[frame_no, :, :, :])

        return np.uint8(vessel_map/no_frames)

    def convert_collection_to_grayscale(self, image_collection):
        no_frames = image_collection.shape[0]
        height = image_collection.shape[1]
        width = image_collection.shape[2]

        image_collection_grayscale = np.zeros((no_frames, height, width), dtype=np.uint8)
        for frame_no in range(no_frames):
            grayscaled = cv2.cvtColor(image_collection[frame_no, :, :, :], cv2.COLOR_BGR2GRAY)
            image_collection_grayscale[frame_no, :, :] = grayscaled

        return image_collection_grayscale

    def process_video(self, roi_extraction=False, filter_enabled=False, average_frames=False):
        video = cv2.VideoCapture(self.video_name)
        image_collection = self.extract_video_frames(video)
        video.release()

        if roi_extraction:
            image_collection = self.extract_ROI_from_collection(image_collection)

        image_collection = self.convert_collection_to_grayscale(image_collection)

        if filter_enabled:
            image_collection = ImageProcessor3D().gaussian_filter_3D(image_collection, input_sigma=1, filter_order=0)

        if average_frames:
            image_collection = ImageProcessor3D().create_time_chunks(image_collection, frame_overlap=5, chunk_size=10)

        return image_collection

