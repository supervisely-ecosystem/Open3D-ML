import os
import open3d.ml as _ml3d
import tensorflow as tf
from ml3d.tf.pipelines import ObjectDetection
from ml3d.datasets.sly_dataset import SlyProjectDataset
from ml3d.tf.models.point_pillars import PointPillars
import supervisely_lib as sly
from supervisely_lib.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
import sly_globals as g


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
    remote_model_dir, remote_model_weights_name = os.path.split(g.remote_weights_path)
    remote_model_index = sly.fs.get_file_name(g.remote_weights_path) + '.index'
    remote_config_dir = remote_model_dir
    # Load config ../../../info/*.yml  (assert unique yml in dir)
    for i in range(3):
        remote_config_dir = os.path.split(remote_config_dir)[0]
    remote_config_dir = os.path.join(remote_config_dir, 'info')
    info_file_list = g.api.file.list(g.team_id, remote_config_dir)
    config = [x['name'] for x in info_file_list if x['name'].endswith('yml')]
    assert len(config) == 1
    remote_config_file = os.path.join(remote_config_dir, config[0])

    g.local_weights_path = os.path.join(g.my_app.data_dir, remote_model_weights_name)
    g.local_index_path = os.path.join(g.my_app.data_dir, remote_model_index)
    g.local_model_config_path = os.path.join(g.my_app.data_dir, config[0])

    g.api.file.download(g.team_id, g.remote_weights_path, g.local_weights_path)
    g.api.file.download(g.team_id, os.path.join(remote_model_dir, remote_model_index), g.local_index_path)
    g.api.file.download(g.team_id, remote_config_file, g.local_model_config_path)

    sly.logger.debug(f"Remote weights {g.remote_weights_path}")
    sly.logger.debug(f"Local weights {g.local_weights_path}")
    sly.logger.debug(f"Local index {g.local_index_path}")
    sly.logger.debug(f"Local config path {g.local_model_config_path}")
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
    from ml3d.tf.models.point_pillars_no_norm import PointPillarsNoNorm
    model = PointPillarsNoNorm(**cfg.model)
    pipeline = ObjectDetection(model=model, **cfg.pipeline)
    pipeline.load_ckpt(g.local_ckpt_path)
    return pipeline


def construct_model_meta():
    cfg = _ml3d.utils.Config.load_from_file(g.local_model_config_path)

    labels = cfg.model.classes
    g.gt_index_to_labels = dict(enumerate(labels))
    g.gt_labels = {v: k for k, v in g.gt_index_to_labels.items()}

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection([sly.ObjClass(k, Cuboid3d) for k in labels]))
    sly.logger.info(g.meta.to_json())


def find_unique_file(dir_where, endswith):
    files = [x for x in os.listdir(dir_where) if x.endswith(endswith)]
    if not files:
        sly.logger.error(f'No {endswith} file found in {dir_where}!')
    elif len(files) > 1:
        sly.logger.error(f'More than one {endswith} file found in {dir_where}\n!')
    else:
        return os.path.join(dir_where, files[0])
    return None


@sly.timeit
def deploy_model():
    file = g.local_weights_path
    if os.path.exists(file):
        g.local_ckpt_path = file.split('.')[0]
        g.model = init_model()
        sly.logger.info("Model has been successfully deployed")
    else:
        msg = f"Wrong model path: {file}!"
        sly.logger.error(msg)
        raise ValueError(msg)


def prediction_to_geometries(prediction):
    geometries = []
    for l in prediction:
        bbox = l.to_xyzwhlr()
        dim = bbox[[3, 5, 4]]
        pos = bbox[:3] + [0, 0, dim[1] / 2]
        yaw = bbox[-1]
        position = Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
        rotation = Vector3d(0, 0, float(-yaw))

        dimension = Vector3d(float(dim[0]), float(dim[2]), float(dim[1]))
        geometry = Cuboid3d(position, rotation, dimension)
        geometries.append(geometry)

    return geometries


def prediction_to_annotation(prediction):
    geometries = prediction_to_geometries(prediction)
    figures = []
    objs = []
    for l, geometry in zip(prediction, geometries):  # by object in point cloud
        pcobj = sly.PointcloudObject(g.meta.get_obj_class(l.label_class))
        figures.append(sly.PointcloudFigure(pcobj, geometry))
        objs.append(pcobj)

    annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
    return annotation


def filter_prediction_threshold(predictions, thresh):
    filtered_pred = []
    for bevbox in predictions:
        if bevbox.confidence >= thresh:
            filtered_pred.append(bevbox)
    return filtered_pred


def inference_model(model, local_pointcloud_path, thresh=0.3):
    """Inference 1 pointcloud with the detector.

    Args:
        model (nn.Module): The loaded detector (ObjectDetection pipeline instance).
        local_pointcloud_path: str: The pointcloud filename.
    Returns:
        result Pointcloud.annotation object`.
    """

    pc = SlyProjectDataset.read_lidar(local_pointcloud_path)

    data = {
        'point': pc,
        'feat': None,
        'bounding_boxes': None,
    }

    gen_func, gen_types, gen_shapes = model.model.get_batch_gen([{"data": data}], steps_per_epoch=None, batch_size=1)
    loader = tf.data.Dataset.from_generator(
        gen_func, gen_types,
        gen_shapes).prefetch(tf.data.experimental.AUTOTUNE)

    annotations = []
    # TODO: add confidence to tags
    for data in loader:
        pred = g.model.run_inference(data)

        try:
            pred_by_thresh = filter_prediction_threshold(pred[0], thresh)  # pred[0] because batch_size == 1
            annotation = prediction_to_annotation(pred_by_thresh)
            annotations.append(annotation)
        except Exception as e:
            sly.logger.exception(e)
            raise e
    return annotations[0]  # 0 == no batch inference, loader should return 1 annotation
