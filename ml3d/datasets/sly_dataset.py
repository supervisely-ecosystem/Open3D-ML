import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml
import open3d as o3d
import supervisely_lib as sly
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from .utils import DataProcessing, BEVBox3D

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SlyProjectDataset(BaseDataset):
    def __init__(self, project_path, name='Supervisely', val_split=0, shuffle_seed=None, **kwargs):
        super().__init__(dataset_path=project_path, name=name, val_split=val_split, **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path



        self.project_fs = sly.PointcloudProject.read_single(project_path)
        self.meta = self.project_fs.meta
        assert self.project_fs.total_items - val_split > 0

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names.items())
        for dataset_fs in self.project_fs:
            self.dataset = dataset_fs
            items = list(dataset_fs._item_to_ann)
            # if shuffle_seed:
            #     np.random.seed(shuffle_seed)
            # np.random.shuffle(items)
            # break
        self.train_split = items[val_split:]
        self.val_split = items[:val_split]
        self.test_split = None

    def get_label_to_names(self):
        labels = [x.name for x in self.meta.obj_classes]
        label_to_names = dict(enumerate(labels))
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        pcloud = o3d.io.read_point_cloud(path)
        points = np.asarray(pcloud.points, dtype=np.float32)
        intensity = np.asarray(pcloud.colors, dtype=np.float32)[:, 0:1]
        pc = np.hstack((points, intensity)).astype("float32")

        return pc

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return SlyDatasetSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['train', 'training']:
            return self.train_split
        elif split in ['test', 'testing']:
            return self.test_split
        elif split in ['val', 'validation']:
            return self.val_split
        elif split in ['all']:
            return self.train_split + self.val_split
        else:
            raise ValueError("Invalid split {}".format(split))

    @staticmethod
    def read_label(path, meta):
        ann_json = sly.io.json.load_json_file(path)
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
                           confidence=1.0)

            objects.append(obj)
        return objects

    def is_tested(self):
        pass

    def save_test_result(self):
        pass


class SlyDatasetSplit(BaseDatasetSplit):
    def __init__(self, project, split='train'):
        self.cfg = project.cfg
        self.path_list = project.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(self.__len__(), split))
        self.split = split
        self.project = project
        self.meta = None

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        item = self.path_list[idx]
        item_path, related_images_dir, ann_path = self.project.dataset.get_item_paths(item)
        pc = self.project.read_lidar(item_path)
        label_sly = self.project.read_label(ann_path, self.project.meta)

        data = {
            'point': pc,
            'feat': None,
            'bounding_boxes': label_sly,
        }

        return data

    def get_attr(self, idx):
        item = self.path_list[idx]
        item_path, _, _ = self.project.dataset.get_item_paths(item)

        attr = {'name': item, 'path': item_path, 'split': self.split}
        return attr


DATASET._register_module(SlyProjectDataset)
