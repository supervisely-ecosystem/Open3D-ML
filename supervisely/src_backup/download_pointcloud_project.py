import os
import shutil

import supervisely_lib as sly
from supervisely_lib.project.pointcloud_project import download_pointcloud_project


if __name__ == "__main__":
    dest_dir = '/data/sly_project10'
    shutil.rmtree(dest_dir, ignore_errors=True)  # WARNING!
    project_id = 6572
    api = sly.Api.from_env()

    download_pointcloud_project(api, project_id, dest_dir, dataset_ids=None, download_items=True, log_progress=True)


    sly.logger.info('PROJECT_DOWNLOADED', extra={'dest_dir': dest_dir,
                                                 'datasets': [x for x in os.listdir(dest_dir) if 'json' not in x]})

