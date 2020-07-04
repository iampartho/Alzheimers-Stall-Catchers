import copy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from preprocess_images import ImageProcessor3D


class PointCloud:

    def __init__(self):
        self.rotations = np.array([[0, 0, 0], [np.pi/6, 0, np.pi/6], [np.pi/6, np.pi/6, 0],
                                  [0, np.pi/6, 0], [-np.pi/6, np.pi/6, 0], [-np.pi/4, 0, np.pi/6],
                                  [-np.pi/6, -np.pi/6, 0], [0, -np.pi/6, 0], [np.pi/6, -np.pi/6, 0]])

    def show_cloud(self, cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.visualization.draw_geometries([pcd])

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

    def cloud_projection(self, cloud, depth=50, height=224, width=224):
        voxels = ImageProcessor3D().voxel_grid_from_cloud(cloud, out_depth=depth, out_height=height, out_width=width)
        projection = np.zeros((voxels.shape[1], voxels.shape[2]), dtype=np.uint8)

        contour_pix = 255
        for layer in range(depth):
            projection[np.logical_and(projection == 0, voxels[layer, :, :] == 255)] = contour_pix
            contour_pix = contour_pix - 5
            if contour_pix < 10:
                contour_pix = 10

        # plt.imshow(projection, cmap='gray')
        # plt.show()
        return projection
