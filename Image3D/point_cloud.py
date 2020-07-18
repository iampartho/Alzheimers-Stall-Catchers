import copy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.ndimage import zoom

from preprocess_images import ImageProcessor3D


class PointCloud:

    def __init__(self):
        pi = np.pi
        self.rotations = np.array([[0, 0, 0], [0, -pi/6, pi/4], [pi/4, pi/6, 0],
                                  [-pi/12, -pi/4, -pi/12], [-pi/6, pi/6, -pi/12], [-pi/4, pi/12, pi/6],
                                  [-pi/6, -pi/6, 0], [0, -pi/6, 0], [pi/6, -pi/6, 0]])

    def show_cloud(self, cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.visualization.draw_geometries([pcd])

    def pc2voxel(self, cloud):

        in_depth = np.max(cloud[:, 0])
        in_height = np.max(cloud[:, 1])
        in_width = np.max(cloud[:, 2])
        voxel_grid_original = np.zeros((in_depth + 1, in_height + 1, in_width + 1), dtype=np.uint8)
        voxel_grid_original[cloud[:, 0], cloud[:, 1], cloud[:, 2]] = 255

        return voxel_grid_original

    def pc2voxel_reshaped_depth(self, cloud, depth):

        in_depth = np.max(cloud[:, 0])
        in_height = np.max(cloud[:, 1])
        in_width = np.max(cloud[:, 2])

        voxel_grid_original = np.zeros((in_depth + 1, in_height + 1, in_width + 1), dtype=np.uint8)
        voxel_grid_original[cloud[:, 0], cloud[:, 1], cloud[:, 2]] = 255

        depth_ratio = depth / (in_depth + 1)

        voxel_grid_reshaped = zoom(voxel_grid_original, (depth_ratio, 1, 1))

        return voxel_grid_reshaped

    def rotate(self, cloud, pos):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        center = [np.max(cloud[:, 0]) / 2, np.max(cloud[:, 1]) / 2, np.max(cloud[:, 2]) / 2]

        trans = o3d.geometry.TriangleMesh.create_coordinate_frame()
        pcd_copy = copy.deepcopy(pcd)
        R = trans.get_rotation_matrix_from_xyz((self.rotations[pos, 0], self.rotations[pos, 1], self.rotations[pos, 2]))
        pcd_copy.rotate(R, center=(center[0], center[1], center[2]))

        cloud_rotated = np.asarray(pcd_copy.points)
        cloud_rotated[:, 0] = cloud_rotated[:, 0] - np.min(cloud_rotated[:, 0])
        cloud_rotated[:, 1] = cloud_rotated[:, 1] - np.min(cloud_rotated[:, 1])
        cloud_rotated[:, 2] = cloud_rotated[:, 2] - np.min(cloud_rotated[:, 2])
        cloud_rotated = cloud_rotated.astype(np.uint16)
        return cloud_rotated

    def cloud_projection(self, cloud):
        voxels = self.pc2voxel(cloud)
        projection = np.zeros((voxels.shape[1], voxels.shape[2]), dtype=np.uint8)

        contour_pix = 255
        step = np.floor(contour_pix / voxels.shape[0])
        if step == 0:
            step = 1

        for layer in range(voxels.shape[0]):
            projection[np.logical_and(projection == 0, voxels[layer, :, :] == 255)] = contour_pix
            contour_pix = contour_pix - step
            if contour_pix < 10:
                contour_pix = 10

        return projection


