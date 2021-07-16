# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#import supervisely_lib as sly

import logging
import os.path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from supervisely.src import UploadSly


def download_dataset():
    up = UploadSly(project_id=5268, ds_id=7359)
    up.download_dataset(f"supervisely_dataset")



if __name__ == '__main__':
    #download_dataset()

    from ml3d.datasets.supervisely_object_detection import SuperviselyDetectionDataset
    from ml3d.tf.models.point_pillars_nocalib import PointPillarsNocalib
    from ml3d.tf.pipelines import ObjectDetection
    from ml3d.tf.dataloaders import TFDataloader
    import tqdm
    import open3d._ml3d as _ml3d





    ### TRAIN ###
    # cfg = _ml3d.utils.Config.load_from_file("pointpillars_supervisely.yml")
    # cfg.model.ckpt_path = None
    # model = PointPillarsNocalib(**cfg.model)
    # dataset = SuperviselyDetectionDataset(**cfg.dataset)
    #
    # pipeline = ObjectDetection(model=model, dataset=dataset, **cfg.pipeline)
    # import pprint
    # pipeline.cfg_tb = {
    #     "readme": "readme",
    #     "cmd_line": "cmd_line",
    #     "dataset": pprint.pformat(cfg.dataset, indent=2),
    #     "model": pprint.pformat(cfg.model, indent=2),
    #     "pipeline": pprint.pformat(cfg.pipeline, indent=2),
    # }
    # pipeline.run_train()

    ### EVALUATE ###

    # cfg = _ml3d.utils.Config.load_from_file("pointpillars_supervisely.yml")
    #
    # cfg.model.ckpt_path = "./logs/PointPillarsNocalib_SuperviselyDetectionDataset_tf/checkpoint/ckpt-13"
    # model = PointPillarsNocalib(**cfg.model)
    # dataset = SuperviselyDetectionDataset(**cfg.dataset)
    #
    # pipeline = ObjectDetection(model=model, dataset=dataset, **cfg.pipeline)
    #
    # pipeline.run_test()


    ## INFERENCE ON RAW DATA ###

    cfg = _ml3d.utils.Config.load_from_file("pointpillars_supervisely.yml")


    model = PointPillarsNocalib(**cfg.model)
    pipeline = ObjectDetection(model=model, dataset=None, **cfg.pipeline)
    pipeline.load_ckpt(ckpt_path="./logs/PointPillarsNocalib_SuperviselyDetectionDataset_tf/checkpoint/ckpt-14")
    rawloader = SuperviselyDetectionDataset.get_input_loader("/data/app_data/supervisely_dataset/000001.pcd")

    process_bar = tqdm.tqdm(rawloader, total=1, desc='inference')
    pred = []
    for data in process_bar:
        results = pipeline.run_inference(data)
        pred.append(results[0])

    for p in pred:
        filtered = []
        for bevbox3d in p:
            if bevbox3d.confidence > 0.6:
                filtered.append((bevbox3d.label_class, bevbox3d.center))
        print("result", filtered)


    print("FINISHED!")
    exit(1)




    # ### INFERENCE ON DATASET###
    # cfg = _ml3d.utils.Config.load_from_file("pointpillars_supervisely.yml")
    # model = PointPillarsNocalib(**cfg.model)
    # dataset = SuperviselyDetectionDataset(**cfg.dataset)
    #
    #
    # pipeline = ObjectDetection(model=model, dataset=dataset, **cfg.pipeline)
    #
    # pipeline.load_ckpt(ckpt_path="./logs/PointPillarsNocalib_SuperviselyDetectionDataset_tf/checkpoint/ckpt-13")
    # test_split = dataset.get_split('test')
    # data_loader = TFDataloader(dataset=test_split,
    #                           model=model,
    #                           use_cache=False)
    # test_loader, len_test = data_loader.get_loader(1, transform=False)
    # pred = []
    # len_test = 2
    # i = 0
    # process_bar = tqdm.tqdm(test_loader, total=len_test, desc='inference')
    # for data in process_bar:
    #     if i >= len_test:
    #         break
    #     results = pipeline.run_inference(data)
    #     pred.append(results[0])
    #     i += 1
    #
    # for p in pred:
    #     filtered = []
    #     for bevbox3d in p:
    #         if bevbox3d.confidence > 0.6:
    #             filtered.append((bevbox3d.label_class, bevbox3d.center))
    #     print("result", filtered)
    #
    # gt = test_split.get_data(0)
    # print("gt", [(x.label_class, x.center) for x in gt['bounding_boxes']])
    # gt = test_split.get_data(1)
    # print("gt", [(x.label_class, x.center) for x in gt['bounding_boxes']])
    #
    # print("FINISHED!")
    # exit(1)


    ### DEFAULT INFERENCE ON KITTI###
    # import open3d.ml as _ml3d
    # import open3d.ml.tf as ml3d
    #
    # from ml3d.tf.dataloaders import TFDataloader
    # cfg_file = "../../ml3d/configs/pointpillars_kitti.yml"
    # cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    #
    # model = ml3d.models.PointPillars(**cfg.model)
    # cfg.dataset['dataset_path'] = "/data/KITTI_object_detection_example"
    # dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    # pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)
    #
    # ckpt_path = "/data/pointpillars_kitti/ckpt-12"
    # pipeline.load_ckpt(ckpt_path=ckpt_path)
    #
    # test_split = dataset.get_split("train")
    #
    # test_split = TFDataloader(dataset=test_split,
    #                           model=model,
    #                           use_cache=False)
    #
    # test_loader, len_test = test_split.get_loader(1,
    #                                               transform=False)
    #
    # import tqdm
    # pred = []
    # process_bar = tqdm.tqdm(test_loader, total=len_test, desc='inference')
    # for data in process_bar:
    #     results = pipeline.run_inference(data)
    #     pred.append(results[0])






