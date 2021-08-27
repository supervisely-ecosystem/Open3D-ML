import supervisely_lib as sly
import tqdm
import shutil
from supervisely_lib.project.pointcloud_project import OpenMode
from supervisely_lib.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
import numpy as np
from ml3d.datasets.sly_dataset import SlyProjectDataset
from supervisely_lib.geometry.cuboid_3d import Cuboid3d


class PointCloudFilter:
    def __call__(self, ann, pointcloud):
        new_objs = []
        new_figures = []
        for fig in ann.figures:
            obj = fig.parent_object
            if self._check_func(obj, fig, pointcloud):
                new_figures.append(fig)
                new_objs.append(obj)

        if new_objs:
            new_annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(new_objs), new_figures)
        else:
            new_annotation = None

        return new_annotation, pointcloud

    def _check_func(self, obj, fig, pointcloud):
        return True


class RangeFilter(PointCloudFilter):
    def __init__(self, max_range):
        self.max_range = max_range  # [xmin ymin zmin xmax ymax zmax]

    def _check_func(self, obj, fig, pointcloud):
        xmin, ymin, zmin, xmax, ymax, zmax = self.max_range
        pos = fig.geometry.position

        return (pos.x > xmin) & (pos.y > ymin) & (pos.z > zmin) & \
               (pos.x < xmax) & (pos.y < ymax) & (pos.z < zmax)


class MinPointsFilter(PointCloudFilter):
    def __init__(self, min_points):
        self.min_points = min_points

    def _check_func(self, obj, fig, pointcloud):
        from ml3d.datasets.utils.operations import points_in_box
        pos = fig.geometry.position  # Maybe bug: need shift or not (?)
        dim = fig.geometry.dimensions
        yaw = fig.geometry.rotation.z

        bbox = [pos.x, pos.y, pos.z - dim.z / 2, dim.x, dim.y, dim.z, yaw]

        # bbox[0:3] = [pos.x, pos.y, pos.z] - [0, 0, dim.z / 2]

        p = points_in_box(pointcloud, rbbox=[bbox])  # too slow :(
        points_count_inside_box = sum(p.flatten())
        return points_count_inside_box > self.min_points


class ClassRemapFilter(PointCloudFilter):
    def __init__(self, class_map):
        self.class_map = class_map
        self.meta = self._create_meta_from_classmap()

    def __call__(self, ann, pointcloud):
        new_objs = []
        new_figures = []
        for fig in ann.figures:
            obj = fig.parent_object
            if obj.obj_class.name in self.class_map.keys():
                new_class_name = self.class_map[obj.obj_class.name]
                pcobj = sly.PointcloudObject(self.meta.get_obj_class(new_class_name))
                new_figures.append(sly.PointcloudFigure(pcobj, fig.geometry))
                new_objs.append(pcobj)

        if new_objs:
            new_annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(new_objs), new_figures)
        else:
            new_annotation = None
        return new_annotation, pointcloud

    def _create_meta_from_classmap(self):
        labels = set(self.class_map.values())
        obj_classes = [sly.ObjClass(k, Cuboid3d) for k in labels]
        meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        return meta

    def _check_func(self, obj, fig, pointcloud):
        raise NotImplementedError


class PointCloudFilterByNames(PointCloudFilter):
    def __init__(self, name_list, allow=True):
        """
        :param name_list: list [ str: item_name, str: item_name, ...]
            regular item_name returned by sly.dataset
        :param allow: whitelist mode if true, blacklist if false.
        """
        self.allow = allow
        self.name_list = name_list  # str: item_name

    def __call__(self, ann, pointcloud):
        if ann.description in self.name_list and self.allow:
            return ann, pointcloud

        elif ann.description not in self.name_list and not self.allow:
            return ann, pointcloud

        else:
            return None, None


def create_meta_from_annotations(annotations):
    objs = []
    for ann in annotations:
        objs.extend(ann.objects)

    meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection({obj.obj_class for obj in objs}))
    return meta


def apply_to_project(funcs, project_dir):
    project_fs = sly.PointcloudProject.read_single(project_dir)

    new_project_dir = project_dir + '_filtered'
    shutil.rmtree(new_project_dir, ignore_errors=True)
    project_filtered = sly.PointcloudProject(new_project_dir, OpenMode.CREATE)
    project_filtered.set_meta(project_fs.meta)  # set tmp meta

    for dataset_fs in project_fs:
        dataset_filtered = project_filtered.create_dataset(dataset_fs.name)
        anns = []
        for item_name in tqdm.tqdm(dataset_fs, total=len(dataset_fs)):
            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            ann_json = sly.io.json.load_json_file(ann_path)
            ann_json['description'] = item_name
            ann = sly.PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            pointcloud = SlyProjectDataset.read_lidar(item_path)

            for func in funcs:
                if ann:
                    ann, pointcloud = func(ann, pointcloud)
                else:
                    break

            if ann:
                anns.append(ann)
                new_item_path = dataset_filtered.generate_item_path(item_name)
                shutil.copy(item_path, new_item_path)
                dataset_filtered.add_item_file(item_name, new_item_path, ann)
        if anns:
            new_meta = create_meta_from_annotations(anns)
            project_filtered.set_meta(new_meta)
            print(new_meta)
            print("New project created!", new_project_dir)
        else:
            print("No annotations after filtering")
            shutil.rmtree(new_project_dir)


ignore = PointCloudFilterByNames(['host-a101_lidar1_1241889714302424374.pcd',
                                  'host-a011_lidar1_1233090647501149366.pcd'])  # only 2 pointcloud
rf = RangeFilter([-40,-40,-10, 40, 40,10])
mf = MinPointsFilter(50)
cf = ClassRemapFilter({"car": "car",
                       "truck": "truck",
                      })

apply_to_project([cf, rf, mf], "/data/LyftSequence4")

