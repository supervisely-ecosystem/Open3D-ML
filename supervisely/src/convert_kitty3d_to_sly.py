import glob
import os

import numpy as np
import open3d as o3d
import supervisely_lib

import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely_lib.project.pointcloud_project import PointcloudProject, OpenMode, ProjectMeta
import logging

logger = logging.getLogger()

import shutil


kitti_dataset_path = "/data/kitti_with_images"
sly_project_path = "/data/converted_kitti_project3"
sly_dataset_name = "sly_converted_from_kitti"

# tmp_file = os.environ["DEBUG_APP_DIR"] + "/pointcloud.pcd"

shutil.rmtree(sly_project_path, ignore_errors=True)

def get_kitti_files_list(kitti_dataset_path):
    binfiles_glob = os.path.join(kitti_dataset_path, "velodyne/*.bin")
    bin_paths = sorted(glob.glob(binfiles_glob))
    if len(bin_paths) < 1:
        sly.logger.error(f"No pointclouds found! Check path: {binfiles_glob}")
    label_paths = [x.replace('velodyne', 'label_2').replace('.bin', '.txt') for x in bin_paths]
    image_paths = [x.replace('velodyne', 'image_2').replace('.bin', '.png') for x in bin_paths]
    calib_paths = [x.replace('label_2', 'calib') for x in label_paths]
    return bin_paths, label_paths, image_paths, calib_paths

def read_kitti_annotations(label_paths, calib_paths):
    all_labels = []
    all_calib = []
    for label_file, calib_file in zip(label_paths, calib_paths):
        calib = o3d.ml.datasets.KITTI.read_calib(calib_file)
        labels = o3d.ml.datasets.KITTI.read_label(label_file, calib)
        all_labels.append(labels)
        all_calib.append(calib)
    return all_labels, all_calib

def convert_labels_to_meta(labels, geometry=Cuboid3d):
    labels = flatten(labels)
    unique_labels = np.unique([l.label_class for l in labels])
    obj_classes = [sly.ObjClass(k, geometry) for k in unique_labels]
    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
    return meta

def convert_bin_to_pcd(bin_file, save_filepath):
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:, 0:3]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.io.write_point_cloud(save_filepath, pc)

def flatten(list_2d):
    return sum(list_2d, [])

def _convert_label_to_geometry(label):
    geometries = []
    for l in label:
        position = Vector3d(float(l.center[0]), float(l.center[1]), float(l.center[2]))

       # if reverse:
       #     yaw = float(-l.yaw) - np.pi
       #     yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
       # else:
        yaw = -l.yaw

        rotation = Vector3d(0, 0, float(yaw))
        dimension = Vector3d(float(l.size[0]), float(l.size[2]), float(l.size[1]))
        geometry = Cuboid3d(position, rotation, dimension)
        geometries.append(geometry)
    return geometries


def convert_label_to_annotation(label):
    geometries = _convert_label_to_geometry(label)
    figures = []
    objs = []
    for l, geometry in zip(label, geometries):  # by object in point cloud
        pcobj = sly.PointcloudObject(meta.get_obj_class(l.label_class))
        figures.append(sly.PointcloudFigure(pcobj, geometry))
        objs.append(pcobj)

    annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
    return annotation


bin_paths, label_paths, image_paths, calib_paths = get_kitti_files_list(kitti_dataset_path)
kitti_labels, kitti_calibs = read_kitti_annotations(label_paths, calib_paths)

sly.logger.info(f"Loading KITTI dataset with {len(bin_paths)} pointclouds")

# SET Project
project_fs = sly.PointcloudProject(sly_project_path, OpenMode.CREATE)
dataset_fs = project_fs.create_dataset(sly_dataset_name)

meta = convert_labels_to_meta(kitti_labels)
project_fs.set_meta(meta)

def convert_calib_to_image_meta(image_name, calib):

    extrinsicMatrix = [0. for x in range(12)]
    intrinsicMatrix = [0. for x in range(9)]

    data = {
        "name": image_name,
        "meta": {
            "sensorsData": {
                "extrinsicMatrix": extrinsicMatrix,
                "intrinsicMatrix": intrinsicMatrix
            }
        }
    }
    return data

for bin_path, kitti_label, image_path, kitti_calib in zip(bin_paths, kitti_labels, image_paths, kitti_calibs):
    item_name = sly.fs.get_file_name(bin_path) + ".pcd"
    item_path = dataset_fs.generate_item_path(item_name)

    convert_bin_to_pcd(bin_path, item_path) # automatically save pointcloud to itempath
    ann = convert_label_to_annotation(kitti_label)

    dataset_fs.add_item_file(item_name, item_path, ann)

    related_images_path = dataset_fs.get_related_images_path(item_name)
    os.makedirs(related_images_path, exist_ok=True)
    image_name = sly.fs.get_file_name_with_ext(image_path)
    sly_path_img = os.path.join(related_images_path, image_name)
    shutil.copy(src=image_path, dst=sly_path_img)
    img_info = convert_calib_to_image_meta(image_name, kitti_calib)
    sly.json.dump_json_file(img_info, sly_path_img + '.json')


"""
{
  "name": "0000000000.png",
  "meta": {
    "sensorsData": {
      "extrinsicMatrix": [
        0.007533745,
        -0.9999714,
        -0.000616602,
        -0.004069766,
        0.01480249,
        0.0007280733,
        -0.9998902,
        -0.07631618,
        0.9998621,
        0.00752379,
        0.01480755,
        -0.2717806
      ],
      "intrinsicMatrix": [
        721.5377,
        0,
        609.5593,
        0,
        721.5377,
        172.854,
        0,
        0,
        1
      ]
    }
  }
}"""

    #
    # if len(related_items) != 0:
    #     rimg_infos = []
    #     for img_path, meta_json in related_items:
    #         img_name = sly.fs.get_file_name(img_path)
    #
    #         dataset_fs.get_related_images()
    #
    #
    #         img = api.pointcloud.upload_related_image(img_path)[0]
    #         rimg_infos.append({ApiField.ENTITY_ID: pointcloud.id,
    #                            ApiField.NAME: meta_json[ApiField.NAME],
    #                            ApiField.HASH: img,
    #                            ApiField.META: meta_json[ApiField.META]})
    #
    #     api.pointcloud.add_related_images(rimg_infos)









print("DONE!")


exit(1)
# class KittiConverter:
#     def __init__(self, path_to_kitti_dataset):
#         local_dataset_path = path_to_kitti_dataset
#         assert os.path.isdir(local_dataset_path)
#
#         self.tmp_file = os.environ["DEBUG_APP_DIR"] + "/pointcloud.pcd"
#         self.bin_path = glob.glob(local_dataset_path + "velodyne/*.bin")
#
#
#         self.files = zip(self.bin_path, label_path, calib_path)
#         self.meta, self.annotations = self.convert_labels()
#
#         self.size = len(self.bin_path)
#         logger.info(f"Loaded KITTI dataset with {self.size} pointclouds")
#         logger.info(self.meta)
#         self._index = 0
#
#     def __next__(self):
#         if self._index < self.size:
#             name = os.path.basename(self.bin_path[self._index]).replace('.bin', '.pcd')
#             self.tmp_file = os.path.join(os.environ["DEBUG_APP_DIR"], name)
#             self.convert_bin_to_pcd(self.bin_path[self._index], self.tmp_file)
#             anns = self.annotations[self._index]
#             self._index += 1
#
#             return self.tmp_file, anns, name
#         raise StopIteration
#
#     def read_kitty_labels(self):
#         """
#             read whole dataset labels, returns 2d list e.g. [labels_for_pc1, labels_for_pc2... ]
#         """
#         all_labels = []
#         all_calib = []
#         for bin_file, label_file, calib_file in self.files:
#             calib = o3d.ml.datasets.KITTI.read_calib(calib_file)
#             labels = o3d.ml.datasets.KITTI.read_label(label_file, calib)
#             all_labels.append(labels)
#             all_calib
#         return all_labels
#
#     def convert_labels(self):
#         annotations = []
#
#         all_labels = self.read_kitty_labels()
#         meta = self._collect_meta(self.flatten(all_labels))
#
#         for labels in all_labels:  # by point clouds
#             geometry_list = self.kitti_to_sly_geometry(labels)
#
#             figures = []
#             objs = []
#             for l, geometry in zip(labels, geometry_list):  # by object in point cloud
#                 pcobj = sly.PointcloudObject(meta.get_obj_class(l.label_class))
#                 figures.append(sly.PointcloudFigure(pcobj, geometry))
#                 objs.append(pcobj)
#
#             pc_annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
#             annotations.append(pc_annotation)
#
#         return meta, annotations
#
#     def get_meta(self):
#         return self.meta
#
#     @staticmethod
#     def kitti_to_sly_geometry(labels, reverse=True):
#         geometry = []
#         for l in labels:
#             position = Vector3d(float(l.center[0]), float(l.center[1]), float(l.center[2]))
#
#             if reverse:
#                 yaw = float(-l.yaw) - np.pi
#                 yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
#             else:
#                 yaw = -l.yaw
#
#             rotation = Vector3d(0, 0, float(yaw))
#             dimension = Vector3d(float(l.size[0]), float(l.size[2]), float(l.size[1]))
#             g = Cuboid3d(position, rotation, dimension)
#             geometry.append(g)
#         return geometry
#
#     @staticmethod
#     def _collect_meta(labels, geometry=Cuboid3d):
#         """
#         :param labels: list of red KITTI labels
#         :return: sly.ProjectMeta
#         """
#         unique_labels = np.unique([l.label_class for l in labels])
#         obj_classes = [sly.ObjClass(k, geometry) for k in unique_labels]
#         meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
#         return meta
#
#     @staticmethod
#
#
#
#
#
#
#
#
#
# def download_pointcloud_project(api, project_id, dest_dir, dataset_ids=None, download_items=True, log_progress=False):
#     LOG_BATCH_SIZE = 1
#
#     key_id_map = KeyIdMap()
#
#     project_fs = PointcloudProject(dest_dir, OpenMode.CREATE)
#
#     meta = ProjectMeta.from_json(api.project.get_meta(project_id))
#     project_fs.set_meta(meta)
#
#     datasets_infos = []
#     if dataset_ids is not None:
#         for ds_id in dataset_ids:
#             datasets_infos.append(api.dataset.get_info_by_id(ds_id))
#     else:
#         datasets_infos = api.dataset.get_list(project_id)
#
#     for dataset in datasets_infos:
#         dataset_fs = project_fs.create_dataset(dataset.name)
#         pointclouds = api.pointcloud.get_list(dataset.id)
#
#         ds_progress = None
#         if log_progress:
#             ds_progress = Progress('Downloading dataset: {!r}'.format(dataset.name), total_cnt=len(pointclouds))
#         for batch in batched(pointclouds, batch_size=LOG_BATCH_SIZE):
#             pointcloud_ids = [pointcloud_info.id for pointcloud_info in batch]
#             pointcloud_names = [pointcloud_info.name for pointcloud_info in batch]
#
#             ann_jsons = api.pointcloud.annotation.download_bulk(dataset.id, pointcloud_ids)
#
#             for pointcloud_id, pointcloud_name, ann_json in zip(pointcloud_ids, pointcloud_names, ann_jsons):
#                 if pointcloud_name != ann_json[ApiField.NAME]:
#                     raise RuntimeError("Error in api.video.annotation.download_batch: broken order")
#
#                 pointcloud_file_path = dataset_fs.generate_item_path(pointcloud_name)
#                 if download_items is True:
#                     api.pointcloud.download_path(pointcloud_id, pointcloud_file_path)
#
#                     related_images_path = dataset_fs.get_related_images_path(pointcloud_name)
#                     related_images = api.pointcloud.get_list_related_images(pointcloud_id)
#                     for rimage_info in related_images:
#                         name = rimage_info[ApiField.NAME]
#                         rimage_id = rimage_info[ApiField.ID]
#
#                         path_img = os.path.join(related_images_path, name)
#                         path_json = os.path.join(related_images_path, name + ".json")
#
#                         api.pointcloud.download_related_image(rimage_id, path_img)
#                         dump_json_file(rimage_info, path_json)
#
#                 else:
#                     touch(pointcloud_file_path)
#
#                 dataset_fs.add_item_file(pointcloud_name,
#                                          pointcloud_file_path,
#                                          ann=PointcloudAnnotation.from_json(ann_json, project_fs.meta, key_id_map),
#                                          _validate_item=False)
#             if log_progress:
#                 ds_progress.iters_done_report(len(batch))
#
#     project_fs.set_key_id_map(key_id_map)
#
#
