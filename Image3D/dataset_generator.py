import csv
import glob
import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
import torch

from preprocess_images import VideoProcessor, ImageProcessor3D
from point_cloud import PointCloud

class Generator:

    def __init__(self):
        self.data = []

    def extract_name(self, filename, path, suffix):
        # extracts exact filename from path
        name = filename.replace(path, "")
        name = name.replace(suffix, "")
        return name

    def generate_processed_dataset(self, src_path, dst_path, testing=False):
        src_suffix = ".mp4"
        dst_suffix = ".jpg"

        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for f in tqdm(files):
            # Iterates through all the video files and export images
            sub_directory = dst_path + "/" + self.extract_name(f, src_path, src_suffix)
            if not os.path.exists(sub_directory):   # Create a directory for each video file
                os.mkdir(sub_directory)

            extractor = VideoProcessor(f)
            processed_images = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

            for frame_no in range(processed_images.shape[0]):
                output_filename = sub_directory + "/" + str(frame_no) + dst_suffix
                cv2.imwrite(output_filename, processed_images[frame_no, :, :])

            if testing == True:
                break

    def generate_point_cloud_dataset(self, src_path, dst_path, testing=False):
        src_suffix = ".mp4"
        dst_suffix = ".h5"

        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for f in tqdm(files):
            # Iterates through all the video files and export images to file

            extractor = VideoProcessor(f)
            image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

            thresh1 = int(np.percentile(image_collection.ravel(), 99))
            thresh2 = int(np.percentile(image_collection.ravel(), 95))
            thresh3 = int(np.percentile(image_collection.ravel(), 90))

            cloud1, nil = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh1,
                                                                           filter_outliers=True)
            cloud2, nil = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh2,
                                                                           filter_outliers=True)
            cloud3, nil = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh3,
                                                                           filter_outliers=True)

            sample_name = self.extract_name(f, src_path, src_suffix)

            output_filename = dst_path + "/" + sample_name + dst_suffix
            hf = h5py.File(output_filename, 'w')
            hf.create_dataset('cloud1', data=cloud1)
            hf.create_dataset('cloud2', data=cloud2)
            hf.create_dataset('cloud3', data=cloud3)
            hf.close()

            if testing == True:
                break

    def generate_tensor_dataset(self, src_path, dst_path, testing=False):
        src_suffix = ".mp4"
        dst_suffix = ".pt"
        depth, height, width = 32, 64, 64
        percentiles = [99, 95, 90]

        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for f in tqdm(files):
            # Iterates through all the video files and export images to file

            extractor = VideoProcessor(f)
            image_collection = extractor.process_video(roi_extraction=True, filter_enabled=True, average_frames=True)

            in_depth = image_collection.shape[0]
            in_height = image_collection.shape[1]
            in_width = image_collection.shape[2]

            output_3D = np.zeros((3, depth, height, width))

            for i in range(3):
                thresh = int(np.percentile(image_collection.ravel(), percentiles[i]))

                cloud, _ = ImageProcessor3D().point_cloud_from_collecton(image_collection, threshold=thresh,
                                                                        filter_outliers=True)

                mask = np.zeros((in_depth, in_height, in_width)).astype(float)
                mask[cloud[:, 0], cloud[:, 1], cloud[:, 2]] = 1.0
                mask[mask != 1.0] = 0.2

                masked = np.multiply(image_collection.astype(float), mask)
                masked = masked / np.max(masked.ravel())

                depth_ratio = depth / in_depth
                height_ratio = height / in_height
                width_ratio = width / in_width

                output_3D[i, :, :, :] = zoom(masked, (depth_ratio, height_ratio, width_ratio))

            sample_name = self.extract_name(f, src_path, src_suffix)

            output_filename = dst_path + "/" + sample_name + dst_suffix
            output_3D = torch.from_numpy(output_3D).float()
            torch.save(output_3D, output_filename)

            if testing == True:
                break

    def voxel_grid_from_csv(self, src_path, labels_file, shape):
        src_suffix = ".h5"
        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        with open(labels_file, mode='r') as infile:
            reader = csv.reader(infile)
            labels = {rows[0]: rows[1] for rows in reader}
            infile.close()

        total_len = len(files)

        np.random.seed(100)
        randomized_index = np.arange(total_len)
        np.random.shuffle(randomized_index)

        layers = shape[0]
        depth = shape[1]
        height = shape[2]
        width = shape[3]

        X = np.zeros((total_len, layers, depth, height, width))
        y = np.zeros(total_len)

        sample_no = 0
        for f in tqdm(files):
            hf = h5py.File(f, 'r')
            cloud1 = hf['cloud1'][:]
            cloud2 = hf['cloud2'][:]
            cloud3 = hf['cloud3'][:]
            hf.close()

            sample_name = self.extract_name(f, src_path, src_suffix)

            label = labels[sample_name + ".mp4"]

            index = randomized_index[sample_no]
            X[index, 0, :, :, :] = ImageProcessor3D().voxel_grid_from_cloud(cloud1, out_depth=depth, out_height=height, out_width=width)
            X[index, 1, :, :, :] = ImageProcessor3D().voxel_grid_from_cloud(cloud2, out_depth=depth, out_height=height, out_width=width)
            X[index, 2, :, :, :] = ImageProcessor3D().voxel_grid_from_cloud(cloud3, out_depth=depth, out_height=height, out_width=width)
            y[index] = int(label)

            sample_no = sample_no + 1

        return X, y

    def projections_from_cloud(self, src_path, dst_path, testing=False):
        src_suffix = ".h5"
        dst_suffix = ".jpg"

        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for f in tqdm(files):
            sample_name = self.extract_name(f, src_path, src_suffix)
            sub_directory = dst_path + "/" + sample_name
            if not os.path.exists(sub_directory):   # Create a directory for each video file
                os.mkdir(sub_directory)

            hf = h5py.File(f, 'r')
            cloud = hf['cloud2'][:]
            hf.close()

            for i in range(9):
                if i == 0:
                    projection = PointCloud().cloud_projection(cloud=cloud)
                else:
                    cloud_rotated = PointCloud().rotate(cloud=cloud, pos=i)
                    projection = PointCloud().cloud_projection(cloud=cloud_rotated)

                # projection = cv2.resize(projection, (224, 224), interpolation=cv2.INTER_AREA)
                output_filename = sub_directory + "/" + str(i) + dst_suffix
                cv2.imwrite(output_filename, projection)

            if testing == True:
                break

    def images_from_cloud(self, src_path, dst_path, testing=False):
        src_suffix = ".h5"
        dst_suffix = ".jpg"

        files = [f for f in glob.glob(src_path + "*" + src_suffix, recursive=True)]

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        for f in tqdm(files):
            sample_name = self.extract_name(f, src_path, src_suffix)
            sub_directory = dst_path + "/" + sample_name
            if not os.path.exists(sub_directory):   # Create a directory for each video file
                os.mkdir(sub_directory)

            hf = h5py.File(f, 'r')
            cloud = hf['cloud2'][:]
            hf.close()

            frames = PointCloud().pc2voxel_reshaped_depth(cloud, depth=20)

            for i in range(frames.shape[0]):
                frame = frames[i, :, :]
                output_filename = sub_directory + "/" + str(i) + dst_suffix
                cv2.imwrite(output_filename, frame)

            if testing == True:
                break


if __name__ == "__main__":
    # Location of original dataset
    src_directory = "../../dataset/test/"

    # Target Location of converted dataset
    dst_directory = "../../test_tensor"

    # Generator().generate_processed_dataset(src_directory, dst_directory, testing=True)
    # Generator().generate_point_cloud_dataset(src_directory, dst_directory, testing=False)
    # Generator().generate_tensor_dataset(src_directory, dst_directory, testing=False)
    # Generator().projections_from_cloud(src_directory, dst_directory, testing=False)
    # Generator().images_from_cloud(src_directory, dst_directory, testing=False)
    # os.system('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')


    print("Finished")
