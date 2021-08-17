
from ml3d.utils.config import Config
from ml3d.tf.models import PointPillars
from ml3d.datasets.sly_dataset import SlyProjectDataset
from ml3d.tf.pipelines import ObjectDetection
import pprint

cfg = Config.load_from_file("../train/configs/pointpillars_kitti_sly.yml")

model = PointPillars(**cfg.model)

dataset = SlyProjectDataset("/data/sly_project", val_split=5)

pipeline = ObjectDetection(model, dataset, **cfg.pipeline)
pipeline.load_ckpt("/data/pointpillars_kitti_202012221652utc/ckpt-12") #  Pretrained
#pipeline.load_ckpt("/data/INFERENCE_CKPT/ckpt-12")

# TRAIN
pipeline.cfg_tb = {
    "readme": "readme",
    "cmd_line": "cmd_line",
    "dataset": pprint.pformat(cfg.dataset, indent=2),
    "model": pprint.pformat(cfg.model, indent=2),
    "pipeline": pprint.pformat(cfg.pipeline, indent=2),
}

#pipeline.run_train()

# EVAL
pipeline.run_valid()

# INFERENCE
# local_pointcloud_path = "/data/sly_project_kitti/training/velodyne/000002.bin"
# calib_path = "/data/sly_project_kitti/training/calib/000002.txt"
# label_path = "/data/sly_project_kitti/training/label_2/000002.txt"
#
# import open3d as o3d
# import numpy as np
# import tensorflow as tf
# def read_pcd(local_pointcloud_path):
#     pcloud = o3d.io.read_point_cloud(local_pointcloud_path)
#     # R = pcloud.get_rotation_matrix_from_xyz((0, 0, np.pi))
#     # center = pcloud.get_center()
#     # pcloud = pcloud.rotate(R, center)
#     points = np.asarray(pcloud.points, dtype=np.float32)
#     intensity = np.asarray(pcloud.colors, dtype=np.float32)[:, 0:1]
#     pc = np.hstack((points, intensity)).astype("float32")
#     return pc

#
# #pc = read_pcd(local_pointcloud_path) # for pcd
# pc = KITTI.read_lidar(local_pointcloud_path)
# #read_kitti_lidar = np.fromfile(local_pointcloud_path, dtype=np.float32).reshape(-1, 3)
# #print(read_kitti_lidar)
#
# calib = KITTI.read_calib(calib_path)
# label = KITTI.read_label(label_path, calib)
#
# calib = "HELLOWORLD"
# # reduced_pc = DataProcessing.remove_outside_points(
# #     pc, calib['world_cam'], calib['cam_img'], [375, 1242])  # TODO: is it necessary?
#
# data = {
#     'point': pc,
#     # 'full_point': pc,
#     'feat': None,
#     'bounding_boxes': None,
# }
#
# gen_func, gen_types, gen_shapes = model.get_batch_gen([{"data": data}], steps_per_epoch=None, batch_size=1)
# loader = tf.data.Dataset.from_generator(
#     gen_func, gen_types,
#     gen_shapes).prefetch(tf.data.experimental.AUTOTUNE)
#
# annotations = []
# # TODO: add confidence to tags
# for data in loader:
#     print("GO")
#     pred = pipeline.run_inference(data)
#     for i in pred[0]:
#         print(np.mean(i.to_xyzwhlr()))

