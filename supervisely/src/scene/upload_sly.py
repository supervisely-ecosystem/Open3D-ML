import supervisely_lib as sly
import logging
import tqdm
import os
import numpy as np

logger = logging.getLogger()


class UploadSly:
    def __init__(self, project_id=None, project_name='KITTI_import', ds_name='pointcloud', ds_id=None):
        """
            If ds_id is not None: read existing dataset by ds_id
            Else: using ds_name to create new dataset
        """
        self.api = sly.Api.from_env()
        if project_id:
            self.project = self.api.project.get_info_by_id(project_id)
        else:
            self.project = self.api.project.create(os.environ["context.workspaceId"],
                                                   project_name,
                                                   type=sly.ProjectType.POINT_CLOUDS,
                                                   change_name_if_conflict=True)

        if ds_id:
            self.dataset = self.api.dataset.get_info_by_id(ds_id)
            logger.info(f"Api works with existing dataset {self.dataset.name}")
        else:
            self.dataset = self.api.dataset.create(self.project.id, f'{ds_name}', change_name_if_conflict=True)
            logger.info(f"Api create new dataset: {self.dataset.name}")

        logger.info(f"Api works with project {self.project.id} dataset {self.dataset.name}")
        pclouds = self.api.pointcloud.get_list(self.dataset.id)
        self.pclouds_dict = {pc.name: pc for pc in pclouds}
        self.pcloud = None  # current pointcloud can be set later
        logger.info(f"Found {len(pclouds)} pointclouds")

    def upload_annotation(self, annotation):
        self.api.pointcloud.annotation.append(self.pcloud.id, annotation)  # annotation upload
        logger.info(f'Annotation uploaded')

    def update_meta(self, meta):
        self.api.project.update_meta(self.project.id, meta.to_json())
        logger.info(f'Meta udpdated')

    def upload(self, pointcloud_filepaths, annotation, name):
        upload_info = self.api.pointcloud.upload_path(self.dataset.id, name=name, path=pointcloud_filepaths)
        self.api.pointcloud.annotation.append(upload_info.id, annotation)
        logger.info(f'{name} uploaded')

    def set_pcloud(self, cloud_name):
        self.pcloud = self.pclouds_dict[cloud_name]

    def upload_annotation(self, annotation):
        # For prediction load to existing dataset. set pcloud first
        self.api.pointcloud.annotation.append(self.pcloud.id, annotation)  # annotation upload
        logger.info(f'Annotation uploaded')

    def download_dataset(self, dataset_dirname):
        save_path = os.path.join(os.environ['DEBUG_APP_DIR'], dataset_dirname)

        if os.path.isdir(save_path):
            logger.info(f'Dataset {save_path} existed locally, stopped.')
            return
        else:
            os.mkdir(save_path)

        obj_class_infos = self.api.object_class.get_list(self.project.id)
        obj_class_infos = {x.id: x.name for x in obj_class_infos}

        all_bboxes = []
        all_labels = []
        pcloud_paths = []


        for k, v in tqdm.tqdm(self.pclouds_dict.items()):
            data = self.api.pointcloud.annotation.download(v.id)

            self.pcloud = self.pclouds_dict[v.name]
            pcloud_path = os.path.join(save_path, self.pcloud.name)
            self.api.pointcloud.download_path(self.pcloud.id, pcloud_path)

            pcloud_paths.append(pcloud_path)

            figures = data['figures']
            objects = data['objects']
            objects = {x['id']: x for x in objects}

            bboxes = []
            labels = []

            for figure in figures:
                geometry = figure['geometry']
                x, y, z = geometry['position']['x'], geometry['position']['y'], geometry['position']['z']
                w, l, h = geometry['dimensions']['x'], geometry['dimensions']['y'], geometry['dimensions']['z']
                yaw = geometry['rotation']['z']
                class_name = obj_class_infos[objects[figure['objectId']]['classId']]

                bboxes.append([x, y, z, w, l, h, yaw])
                labels.append(class_name)

            all_bboxes.append(bboxes)
            all_labels.append(labels)

        data = np.array([pcloud_paths, all_bboxes, all_labels], dtype=object)
        np.save(os.path.join(save_path, "dataset.npy"), data)
        logger.info(f"Dataset loaded to {save_path}")
