import os
import shutil

import numpy as np
import open3d as o3d
from open3d._ml3d.datasets.utils import BEVBox3D

import supervisely_lib as sly


def kitti_paths(path, mode='write'):
    shutil.rmtree(path, ignore_errors=True)  # WARN!
    os.mkdir(path)
    path = os.path.join(path, 'training')
    bin_dir = os.path.join(path, 'velodyne')
    image_dir = os.path.join(path, 'image_2')
    label_dir = os.path.join(path, 'label_2')
    calib_dir = os.path.join(path, 'calib')

    if mode == 'write' or mode == 'w':
        assert not os.path.exists(path), "Dataset already exists. Remove directory before writing"
        os.mkdir(path)
        os.mkdir(bin_dir)
        os.mkdir(image_dir)
        os.mkdir(label_dir)
        os.mkdir(calib_dir)

    paths = [bin_dir, image_dir, label_dir, calib_dir]
    assert all([os.path.exists(x) for x in paths])
    return paths


def pcd_to_bin(pcd_path, bin_path):
    pcloud = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcloud.points, dtype=np.float32)
    intensity = np.asarray(pcloud.colors, dtype=np.float32)[:,0:1]
    points = np.hstack((points, intensity)).flatten().astype("float32")
    points.tofile(bin_path)


def annotation_to_kitti_label(annotation_path, calib_path, kiiti_label_path, meta):
    ann_json = sly.io.json.load_json_file(annotation_path)
    calib = o3d.ml.datasets.KITTI.read_calib(calib_path)

    ann = sly.PointcloudAnnotation.from_json(ann_json, meta)
    objects = []

    for fig in ann.figures:
        geometry = fig.geometry
        class_name = fig.parent_object.obj_class.name

        dimensions = geometry.dimensions
        position = geometry.position
        rotation = geometry.rotation

        obj = BEVBox3D(center=np.array([float(position.x), float(position.y), float(position.z)]),
                       size=np.array([float(dimensions.x), float(dimensions.z), float(dimensions.y)]),
                       yaw=np.array(float(-rotation.z)),
                       label_class=class_name,
                       confidence=1.0,
                       world_cam=calib['world_cam'],
                       cam_img=calib['cam_img'])

        objects.append(obj)
    if not objects:
        return False
    with open(kiiti_label_path, 'w') as f:
        for box in objects:
            f.write(box.to_kitti_format(box.confidence))
            f.write('\n')
    return True

def mkline(arr):
    return " ".join(map(str, arr))


def write_lines_to_txt(lines, path):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def gen_calib_from_img_meta(img_meta, path):
    extrinsic_matrix = np.array(img_meta['meta']['sensorsData']['extrinsicMatrix'], dtype=np.float32)
    intrinsic_matrix = np.array(img_meta['meta']['sensorsData']['intrinsicMatrix'], dtype=np.float32)

    cam_img = intrinsic_matrix.reshape(3, 3)
    cam_img = np.hstack((cam_img, np.zeros((3, 1)))).flatten()

    empty_line = mkline(np.zeros(12, dtype=np.float32))
    lines = [
        f"P0: {empty_line}",
        f"P1: {empty_line}",
        f"P2: {mkline(cam_img)}",
        f"P3: {empty_line}",
        f"R0_rect: {mkline(np.eye(3, dtype=np.float32).flatten())}",
        f"Tr_velo_to_cam: {mkline(extrinsic_matrix)}",
        f"Tr_imu_to_velo: {empty_line}"]

    write_lines_to_txt(lines, path)



def convert(project_dir, exclude_items=[]):
    kitti_dataset_path = project_dir + "_kitti"
    project_fs = sly.PointcloudProject.read_single(project_dir)
    bin_dir, image_dir, label_dir, calib_dir = kitti_paths(kitti_dataset_path)

    for dataset_fs in project_fs:
        for item_name in dataset_fs:
            if item_name in exclude_items:
                sly.logger.info(f"{item_name} excluded")
                continue

            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)

            item_name_without_ext = item_name.split('.')[0]

            label_path = os.path.join(label_dir, item_name_without_ext + '.txt')
            calib_path = os.path.join(calib_dir, item_name_without_ext + '.txt')
            bin_path = os.path.join(bin_dir, item_name_without_ext + '.bin')
            image_path = os.path.join(image_dir, item_name_without_ext + '.png')


            realted_img_path, img_meta = dataset_fs.get_related_images(item_name)[0]  # ONLY 1 Img

            gen_calib_from_img_meta(img_meta, calib_path)
            labels_exists = annotation_to_kitti_label(ann_path, calib_path=calib_path, kiiti_label_path=label_path, meta=project_fs.meta)
            if not labels_exists:
                continue
            pcd_to_bin(item_path, bin_path)
            shutil.copy(src=realted_img_path, dst=image_path)
            sly.logger.info(f"{item_name} converted to kitti .bin")
    sly.logger.info(f"Dataset converted to kitti and stored at {kitti_dataset_path}")