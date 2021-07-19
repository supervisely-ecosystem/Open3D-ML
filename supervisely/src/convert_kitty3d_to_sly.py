import glob
import os

import numpy as np
import open3d as o3d

import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely_lib.project.pointcloud_project import OpenMode

logger = logging.getLogger()
from collections import namedtuple
import shutil


def get_kitti_files_list(kitti_dataset_path):
    binfiles_glob = os.path.join(kitti_dataset_path, "velodyne/*.bin")
    bin_paths = sorted(glob.glob(binfiles_glob))
    if len(bin_paths) < 1:
        sly.logger.error(f"No pointclouds found! Check path: {binfiles_glob}")
    label_paths = [x.replace('velodyne', 'label_2').replace('.bin', '.txt') for x in bin_paths]
    image_paths = [x.replace('velodyne', 'image_2').replace('.bin', '.png') for x in bin_paths]
    calib_paths = [x.replace('label_2', 'calib') for x in label_paths]
    return bin_paths, label_paths, image_paths, calib_paths



def read_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    objects = []
    for line in lines:
        label = line.strip().split(' ')
        bbox3d = namedtuple("Bbox3D", ("label_class", "center", "size", "yaw"))

        bbox3d.label_class = label[0]
        bbox3d.center = np.array([float(label[11]),
                                 float(label[12]),
                                 float(label[13])])

        bbox3d.size = [float(label[9]), float(label[8]), float(label[10])]
        bbox3d.yaw = float(label[14])
        objects.append(bbox3d)
    return objects

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


def convert_calib_to_image_meta(image_name, calib_path, camera_num=2):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    assert 0 < camera_num < 4
    intrinsic_matrix = lines[camera_num].strip().split(' ')[1:]
    intrinsic_matrix = np.array(intrinsic_matrix, dtype=np.float32).reshape(3, 4)[:3, :3]

    extrinsic_matrix = lines[5].strip().split(' ')[1:]
    extrinsic_matrix = np.array(extrinsic_matrix, dtype=np.float32)

    data = {
        "name": image_name,
        "meta": {
            "sensorsData": {
                "extrinsicMatrix": list(extrinsic_matrix.astype(float)),
                "intrinsicMatrix": list(intrinsic_matrix.flatten().astype(float))
            }
        }
    }
    return data


if __name__ == '__main__':
    kitti_dataset_path = "/data/kitti_with_images"
    sly_project_path = "/data/NoCalibTestConvert"
    sly_dataset_name = "ds22"
    shutil.rmtree(sly_project_path, ignore_errors=True)  # WARN!

    bin_paths, label_paths, image_paths, calib_paths = get_kitti_files_list(kitti_dataset_path)
    kitti_labels, kitti_calibs = read_kitti_annotations(label_paths, calib_paths)

    sly.logger.info(f"Loading KITTI dataset with {len(bin_paths)} pointclouds")

    # SET Project
    project_fs = sly.PointcloudProject(sly_project_path, OpenMode.CREATE)
    dataset_fs = project_fs.create_dataset(sly_dataset_name)

    sly.logger.info(f"Created Supervisely dataset with {dataset_fs.name} at {dataset_fs.directory}")
    meta = convert_labels_to_meta(kitti_labels)
    project_fs.set_meta(meta)
    sly.logger.info(f"Project meta generated:\n{meta}")
    for bin_path, kitti_label, image_path, calib_path in zip(bin_paths, kitti_labels, image_paths, calib_paths):
        item_name = sly.fs.get_file_name(bin_path) + ".pcd"
        item_path = dataset_fs.generate_item_path(item_name)

        convert_bin_to_pcd(bin_path, item_path)  # automatically save pointcloud to itempath
        ann = convert_label_to_annotation(kitti_label)

        dataset_fs.add_item_file(item_name, item_path, ann)

        related_images_path = dataset_fs.get_related_images_path(item_name)
        os.makedirs(related_images_path, exist_ok=True)
        image_name = sly.fs.get_file_name_with_ext(image_path)
        sly_path_img = os.path.join(related_images_path, image_name)
        shutil.copy(src=image_path, dst=sly_path_img)

        img_info = convert_calib_to_image_meta(image_name, calib_path)
        sly.json.dump_json_file(img_info, sly_path_img + '.json')
        sly.logger.info(f".bin -> {item_name}")

    sly.logger.info(f"Job done, dataset converted. Project_path: {sly_project_path}")