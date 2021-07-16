import os

import numpy as np
import open3d as o3d

from scene.base_scene import BaseScene


class KittyScene(BaseScene):
    def __init__(self, path_to_bin_file):
        self.bin_path = path_to_bin_file
        label_path = self.bin_path.replace('velodyne', 'label_2').replace('.bin', '.txt')
        calib_path = label_path.replace('label_2', 'calib')

        self.calib = o3d.ml.datasets.KITTI.read_calib(calib_path)
        self.anns = o3d.ml.datasets.KITTI.read_label(label_path, self.calib)
        self.box_reverse = False

    @property
    def pointcloud(self):
        points = np.fromfile(self.bin_path, dtype=np.float32).reshape(-1, 4)[:, 0:3]
        points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return np.asarray(points.points)

    @property
    def bboxes_3d(self):
        bboxes = []
        for l in self.anns:
            bboxes.append([*l.center, l.size[0], l.size[2], l.size[1], -l.yaw])
        return bboxes

    @property
    def labels(self):
        return [l.label_class for l in self.anns]

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.bin_path))[0]

    @property
    def images(self):
        raise NotImplementedError
