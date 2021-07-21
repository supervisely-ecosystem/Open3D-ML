from ml3d.utils.config import Config
from ml3d.tf.models import PointPillars
from ml3d.datasets import KITTI
from ml3d.tf.pipelines.object_detection import ObjectDetection

import pprint

cfg = Config.load_from_file("pointpillars_kitti_sly.yml")

model = PointPillars(**cfg.model)
dataset = KITTI(**cfg.dataset)

pipeline = ObjectDetection(model, dataset, **cfg.pipeline)
#pipeline.load_ckpt("/data/pointpillars_kitti/ckpt-12") #  Pretrained
#pipeline.load_ckpt("./logs/PointPillars_KITTI_tf/checkpoint/ckpt-2")

# TRAIN
pipeline.cfg_tb = {
    "readme": "readme",
    "cmd_line": "cmd_line",
    "dataset": pprint.pformat(cfg.dataset, indent=2),
    "model": pprint.pformat(cfg.model, indent=2),
    "pipeline": pprint.pformat(cfg.pipeline, indent=2),
}

pipeline.run_train()

# EVAL
#pipeline.run_valid()