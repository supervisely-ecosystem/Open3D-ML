import numpy as np
import os, argparse, pickle, sys

from pathlib import Path
import glob
import logging
import yaml
import open3d as o3d
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from .utils import DataProcessing, BEVBox3D

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SuperviselyDetectionDataset(BaseDataset):
    """This class is used to create a dataset based on the Supervisely dataset, and
    used in object detection, visualizer, training, or testing.
    """

    def __init__(self,
                 dataset_path,
                 name='KITTI',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 val_split=3712,
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (KITTI in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            val_split: The split value to get a set of images for training,
            validation, for testing.
            test_result_folder: Path to store test output.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         val_split=val_split,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 3
        self.label_to_names = self.get_label_to_names()

        self.all_files = glob.glob(cfg.dataset_path + '/*.pcd')
        # self.all_files.sort() # TODO: shuffle
        self.train_files = []
        self.val_files = []

        self.annotations = np.load(os.path.join(cfg.dataset_path, 'dataset.npy'), allow_pickle=True)
        pcloud_paths, bboxes3d, labels = self.annotations
        self.annotations_dict = dict(zip(pcloud_paths, zip(bboxes3d, labels)))

        print(f"{cfg.val_split}, {len(self.all_files)}")
        assert cfg.val_split < len(self.all_files)
        for idx, f in enumerate(self.all_files):
            if idx < cfg.val_split:
                self.train_files.append(f)
            else:
                self.val_files.append(f)

        self.test_files = self.all_files  # TODO: Different testing data!
        self.test_files.sort()

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """
        label_to_names = {
            0: 'Pedestrian',
            1: 'Cyclist',
            2: 'Car',
            3: 'Van',
            4: 'Person_sitting',
            5: 'DontCare'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        pcd = o3d.io.read_point_cloud(path)
        pcd = np.asarray(pcd.points)
        zeros = np.ones(pcd.shape[0]).reshape(-1, 1)
        pcd = np.hstack((pcd, zeros))
        return pcd

    def read_label(self, path):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """



        bboxes3d, labels = self.annotations_dict[path]
        objects = []
        for idx, bbox in enumerate(bboxes3d):
            if labels[idx] == 'DontCare':
                continue
            # bbox from upload_Sly : [x, y, z, w, l, h, yaw]
            center = np.array(
                [float(bbox[0]),
                 float(bbox[1]),
                 float(bbox[2])])

            size = [float(bbox[3]), float(bbox[5]), float(bbox[4])]  # TODO: check it! ( w,h,l ??? )
            yaw = float(bbox[6])
            objects.append(BEVBox3D(center, size, yaw, label_class=labels[idx], confidence=1.0))

        return objects

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate(
            [mat, np.array([[0., 0., 1., 0.]], dtype=mat.dtype)], axis=0)
        return mat

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return SuperviselyDetectionDatasetSplit(self, split=split)

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
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then resturn the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        # make_dir(self.cfg.test_result_folder)
        # for attr, res in zip(attrs, results):
        #     name = attr['name']
        #     path = os.path.join(self.cfg.test_result_folder, name + '.txt')
        #     f = open(path, 'w')
        #     for box in res:
        #         f.write(box.to_kitti_format(box.confidence))
        #         f.write('\n')
        raise NotImplementedError

    @staticmethod
    def get_input_loader(pc_path):
        pc = SuperviselyDetectionDataset.read_lidar(pc_path)

        dataset = {
            'point': pc,
            'feat': None,
            'calib': None,
            'bounding_boxes': None,
        }

        dataset = [{'data': dataset}]
        import tensorflow as tf

        def get_batch_gen(dataset, steps_per_epoch=None, batch_size=1):
            def batcher():
                count = len(dataset) if steps_per_epoch is None else steps_per_epoch
                for i in np.arange(0, count, batch_size):
                    batch = [dataset[i + bi]['data'] for bi in range(batch_size)]
                    points = tf.concat([b['point'] for b in batch], axis=0)

                    count_pts = tf.constant([len(b['point']) for b in batch])
                    no_bboxes = tf.concat([tf.zeros((0, 7), dtype=tf.float32) for b in batch], axis=0)
                    no_labels = tf.concat([tf.zeros((0,), dtype=tf.int32) for b in batch],  axis=0)

                    yield (points, no_bboxes, no_labels, [None], count_pts, no_labels)

            gen_func = batcher
            gen_types = (tf.float32, tf.float32, tf.int32, tf.float32, tf.int32,
                         tf.int32)
            gen_shapes = ([None, 4], [None, 7], [None], [None], [None], [None])

            return gen_func, gen_types, gen_shapes

        gen_func, gen_types, gen_shapes = get_batch_gen(dataset, 1, 1)

        loader = tf.data.Dataset.from_generator(
            gen_func, gen_types,
            gen_shapes).prefetch(tf.data.experimental.AUTOTUNE)

        return loader


class SuperviselyDetectionDatasetSplit:
    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        pc = self.dataset.read_lidar(pc_path)
        label = self.dataset.read_label(pc_path)

        data = {
            'point': pc,
            'feat': None,
            'calib': None,
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr




DATASET._register_module(SuperviselyDetectionDataset)
