import os

import open3d.ml as _ml3d
import tensorflow as tf

from ml3d.tf.pipelines import ObjectDetection

import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d
import globals as g


def _download_dir(remote_dir, local_dir):
    remote_files = g.api.file.list2(g.team_id, remote_dir)
    progress = sly.Progress(f"Downloading {remote_dir}", len(remote_files), need_info_log=True)
    for remote_file in remote_files:
        local_file = os.path.join(local_dir, sly.fs.get_file_name_with_ext(remote_file.path))
        if sly.fs.file_exists(local_file):  # @TODO: for debug
            pass
        else:
            g.api.file.download(g.team_id, remote_file.path, local_file)
        progress.iter_done_report()

@sly.timeit
def download_model_and_configs():
    g.local_weights_path = os.path.join(g.my_app.data_dir, os.path.basename(g.remote_weights_path))
    _download_dir(g.remote_weights_path, g.local_weights_path)
    sly.logger.info("Model has been successfully downloaded")



def init_model():
    cfg = _ml3d.utils.Config.load_from_file(g.local_model_config_path)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if g.device == 'cpu':
                tf.config.set_visible_devices([], 'GPU')
            elif g.device == 'cuda':
                tf.config.set_visible_devices(gpus[0], 'GPU')
            else:
                idx = g.device.split(':')[1]
                tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
        except RuntimeError as e:
            sly.logger.exception(e)

    Model = _ml3d.utils.get_module("model", cfg.model.name, "tf")
    model = Model(**cfg.model)
    pipeline = ObjectDetection(model=model)
    pipeline.load_ckpt(g.local_ckpt_path)
    return pipeline


def construct_model_meta():
    cfg = _ml3d.utils.Config.load_from_file(g.local_model_config_path)

    labels = cfg.model.classes
    g.gt_index_to_labels = dict(enumerate(labels))
    g.gt_labels = {v:k for k,v in g.gt_index_to_labels.items()}

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection([sly.ObjClass(k, Cuboid3d) for k in labels]))
    sly.logger.info(g.meta.to_json())


@sly.timeit
def deploy_model():
    g.local_ckpt_path = os.path.join(g.local_weights_path, os.listdir(g.local_weights_path)[0].split('.')[0])
    g.model = init_model()
    g.model.SLY_CLASSES = sorted(g.gt_labels, key=g.gt_labels.get)
    sly.logger.info("Model has been successfully deployed")



def inference_model(model, pointcloud_local_path):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The pointcloud filename.

    Returns:
        result Pointcloud.annotation object`.
    """
    import open3d as o3d
    import numpy as np
    pcloud = o3d.io.read_point_cloud(pointcloud_local_path)
    points = np.asarray(pcloud.points, dtype=np.float32)
    intensity = np.asarray(pcloud.colors, dtype=np.float32)[:, 0:1]
    points = np.hstack((points, intensity)).flatten().astype("float32")

    data = model.model.preprocess()
    pred = model.run_inference([points, None, None, None])

    return pred


