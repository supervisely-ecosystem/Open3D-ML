import shutil
import pickle
import numpy as np
import open3d as o3d
import supervisely_lib as sly
from supervisely_lib.project.pointcloud_project import OpenMode

from ml3d.datasets.lyft import Lyft
from scripts.preprocess_lyft import LyftProcess
from supervisely.src_backup.convert_kitty3d_to_sly import convert_label_to_annotation, convert_labels_to_meta


def convert_bin_to_pcd(bin_file, save_filepath):
    bin = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
    points = bin[:, 0:3]
    intensity = bin[:, 3]
    ring_index = bin[:, 4]
    intensity_fake_rgb = np.zeros((intensity.shape[0], 3))
    intensity_fake_rgb[:, 0] = intensity  # red The intensity measures the reflectivity of the objects
    intensity_fake_rgb[:, 1] = ring_index  # green ring index is the index of the laser ranging from 0 to 31

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(intensity_fake_rgb)
    o3d.io.write_point_cloud(save_filepath, pc)


def convert(lyft_dataset_path, sly_project_path, sly_dataset_name="ds1"):
    lp = LyftProcess(dataset_path=lyft_dataset_path, out_path=None)
    shutil.rmtree(sly_project_path, ignore_errors=True)  # WARN!

    dataset_info, val_info = lp.process_scenes()  # TODO: not only train!
    sly.logger.info(f"Loading Lyft dataset with {len(dataset_info)} pointclouds")

    # SET Project
    project_fs = sly.PointcloudProject(sly_project_path, OpenMode.CREATE)
    dataset_fs = project_fs.create_dataset(sly_dataset_name)

    sly.logger.info(f"Created Supervisely dataset with {dataset_fs.name} at {dataset_fs.directory}")

    labels = [[Lyft.read_label(info, {'world_cam': None}), info['lidar_path']] for info in dataset_info]
    labels, pc_paths = np.array(labels, dtype=object).T
    meta = convert_labels_to_meta(labels)
    project_fs.set_meta(meta)

    for pc_path, label in zip(pc_paths, labels):
        item_name = sly.fs.get_file_name(pc_path) + ".pcd"
        item_path = dataset_fs.generate_item_path(item_name)

        convert_bin_to_pcd(pc_path, item_path)  # automatically save pointcloud to itempath
        ann = convert_label_to_annotation(label, meta)

        dataset_fs.add_item_file(item_name, item_path, ann)
        sly.logger.info(f".bin -> {item_name}")

    sly.logger.info(f"Job done, dataset converted. Project_path: {sly_project_path}")


def convert_from_pickle(pickle_path, sly_project_path, sly_dataset_name="ds1"):
    shutil.rmtree(sly_project_path, ignore_errors=True)  # WARN!

    with open(pickle_path, 'rb') as f:
        dataset_info = pickle.load(f)

    sly.logger.info(f"Loading Lyft dataset with {len(dataset_info)} pointclouds")

    # SET Project
    project_fs = sly.PointcloudProject(sly_project_path, OpenMode.CREATE)
    dataset_fs = project_fs.create_dataset(sly_dataset_name)

    sly.logger.info(f"Created Supervisely dataset with {dataset_fs.name} at {dataset_fs.directory}")

    labels = [[Lyft.read_label(info, {'world_cam': None}), info['lidar_path']] for info in dataset_info]
    labels, pc_paths = np.array(labels, dtype=object).T
    meta = convert_labels_to_meta(labels)
    project_fs.set_meta(meta)

    for pc_path, label in zip(pc_paths, labels):
        pc_path = "/data/Lyft/v1.01-train/lidar/" + sly.fs.get_file_name_with_ext(pc_path)
        item_name = sly.fs.get_file_name(pc_path) + ".pcd"
        item_path = dataset_fs.generate_item_path(item_name)

        convert_bin_to_pcd(pc_path, item_path)  # automatically save pointcloud to itempath
        ann = convert_label_to_annotation(label, meta)

        dataset_fs.add_item_file(item_name, item_path, ann)
        sly.logger.info(f".bin -> {item_name}")

    sly.logger.info(f"Job done, dataset converted. Project_path: {sly_project_path}")


def cut_scene():
    from lyft_dataset_sdk.lyftdataset import LyftDataset

    import numpy as np
    import pickle
    import os
    lyft = LyftDataset(data_path='/data/Lyft/v1.01-train', json_path='/data/Lyft/v1.01-train/data', verbose=True)
    scene = lyft.scene[0]


    new_token = scene["first_sample_token"]  # pass first frame
    my_sample = lyft.get('sample', new_token)

    dataset_data = []
    for i in range(120):
        new_token = my_sample['next']
        my_sample = lyft.get('sample', new_token)
        lidar_token = my_sample['data']['LIDAR_TOP']
        lidar_path, boxes, _ = lyft.get_sample_data(lidar_token)

        print("SCENE TOKEN", new_token)
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([
            b.orientation.yaw_pitch_roll[0] for b in boxes
        ]).reshape(-1, 1)

        names =  np.array([b.name for b in boxes])
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2],
                                  axis=1)


        data = {
            'lidar_path': lidar_path,
            'gt_boxes': gt_boxes,
            'gt_names': names,
            'num_lidar_pts': [True] * 1000
        }

        dataset_data.append(data)

    with open(os.path.join('/data', 'dataset_data.pkl'), 'wb') as f:
        pickle.dump(dataset_data, f)


if __name__ == "__main__":
    """
        Lyft dataset converter
        Please run "pip install lyft_dataset_sdk" to install the official devkit first.
    """
    # TODO: why intensity == 100?
    lyft_dataset_path = "/data/Lyft"
    sly_project_path = "/data/LyftSequence150"
    sly_dataset_name = "lyft_sample"
#    convert(lyft_dataset_path, sly_project_path, sly_dataset_name)
    import os

    #cut_scene()
    #convert_from_pickle(pickle_path='/data/dataset_data.pkl', sly_project_path=sly_project_path)
