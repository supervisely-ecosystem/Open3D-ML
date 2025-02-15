import os
from pathlib import Path
import sys
import supervisely_lib as sly
import shutil

# for debug
# from dotenv import load_dotenv  # pip install python-dotenv\
# load_dotenv("supervisely/train/debug.env")
# load_dotenv("supervisely/train/secret_debug.env", override=True)
os.environ['OPEN3D_ML_ROOT'] = '/Open3D-ML/set_open3d_ml_root.sh'

inference = False

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
try:
    project_id = int(os.environ['modal.state.slyProjectId'])
except KeyError:
    sly.logger.warning("No project id - this is inference run")
    inference = True

if not inference:
    project_info = api.project.get_info_by_id(project_id)
    if project_info is None:  # for debug
        raise ValueError(f"Project with id={project_id} not found")

    #sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

    project_dir = os.path.join(my_app.data_dir, "sly_project")
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

    artifacts_dir = os.path.join(my_app.data_dir, "artifacts")

    shutil.rmtree(artifacts_dir, ignore_errors=True)
    sly.fs.mkdir(artifacts_dir)
    info_dir = os.path.join(artifacts_dir, "info")
    sly.fs.mkdir(info_dir)
    checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
    sly.fs.mkdir(checkpoints_dir)

    root_source_dir = str(Path(sys.argv[0]).parents[3])
    sly.logger.info(f"Root source directory: {root_source_dir}")
    sys.path.append(root_source_dir)
    source_path = str(Path(sys.argv[0]).parents[0])
    sly.logger.info(f"App source directory: {source_path}")
    sys.path.append(source_path)
    ui_sources_dir = os.path.join(source_path, "ui")
    sly.logger.info(f"UI source directory: {ui_sources_dir}")
    sys.path.append(ui_sources_dir)
    sly.logger.info(f"Added to sys.path: {ui_sources_dir}")
