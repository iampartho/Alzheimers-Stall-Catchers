import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

from preprocess_images import VideoProcessor


class Generator:

    def __init__(self, path, suffix):
        self.data = []
        self.source_path = path
        self.suffix = suffix
        # find all target files in the source folder with given suffixes
        self.files = [f for f in glob.glob(self.source_path + "*" + self.suffix, recursive=True)]

    def print_files(self):
        # print all the filenames in the chosen directory
        for f in self.files:
            print(f)

    def extract_name(self, filename):
        # extracts exact filename from path
        name = filename.replace(self.source_path, "")
        name = name.replace(self.suffix, "")
        return name

    def generate_processed_dataset(self, path="test", suffix=""):

        if not os.path.exists(path):
            os.mkdir(path)

        for f in tqdm(self.files):
            # Iterates through all the video files and export images
            sub_directory = path + "/" + self.extract_name(f)
            if not os.path.exists(sub_directory):   # Create a directory for each video file
                os.mkdir(sub_directory)

            extractor = VideoProcessor(f)
            processed_images = extractor.process_video(roi_extraction=True, average_frames=True)

            for frame_no in range(processed_images.shape[0]):
                output_filename = sub_directory + "/" + str(frame_no) + suffix
                frame = cv2.cvtColor(processed_images[frame_no, :, :, :], cv2.COLOR_BGR2GRAY)
                cv2.imwrite(output_filename, frame)


if __name__ == "__main__":
    # Location of original dataset
    source_directory = "../../micro/"
    source_file_suffix = ".mp4"

    # Target Location of converted dataset
    destination_directory = "../../micro_frames"
    destination_file_suffix = ".jpg"

    # Extract all processed frames from video files
    dataset_generator = Generator(source_directory, source_file_suffix)
    dataset_generator.generate_processed_dataset(destination_directory, destination_file_suffix)

    # Clear PyCache
    os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')
