import logging
import os

import open3d as o3d
import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection

logger = logging.getLogger()


class SlyAnnotation:
    def __init__(self, meta, tmp_dir='/data', uploader=None):
        self.uploader = uploader
        self.scene = None
        self.pc_annotation = None
        self.meta = meta
        self.tmp_dir = tmp_dir

    def get_annotation(self, scene):
        self.set_scene(scene)

        figures = []
        objs = []
        for l, geometry in zip(self.scene.labels, self._geometry):  # by object in point cloud
            pcobj = sly.PointcloudObject(self.meta.get_obj_class(l))
            figures.append(sly.PointcloudFigure(pcobj, geometry))
            objs.append(pcobj)

        self.pc_annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
        return self.pc_annotation

    def set_scene(self, scene):
        self.scene = scene

    def _save_pc(self):
        if not self.scene:
            raise ValueError('Please, set scene first')

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.scene.pointcloud))
        o3d.io.write_point_cloud(self.save_path, pc)
        return self.save_path

    @property
    def save_path(self):
        return f"{self.tmp_dir}{self.scene.name}.pcd"

    @property
    def _geometry(self):
        geometry = []
        for bbox in self.scene.bboxes_3d:
            position = Vector3d(float(bbox[0]), float(bbox[1]), float(bbox[2]))
            dimension = Vector3d(float(bbox[3]), float(bbox[4]), float(bbox[5]))
            rotation = Vector3d(0, 0, float(bbox[6]))
            cuboid = Cuboid3d(position, rotation, dimension)
            geometry.append(cuboid)
        return geometry

    def upload(self):
        if not self.pc_annotation:
            raise ValueError('Please, create annotation first')
        if not self.uploader:
            raise Exception('Please pass the uploader instance')
        self._save_pc()
        self.uploader.upload(self.save_path, self.pc_annotation, self.scene.name)
        os.unlink(self.save_path)
