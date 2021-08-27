import os
import supervisely_lib as sly
import pathlib
import sys

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

os.environ['OPEN3D_ML_ROOT'] = '/Open3D-ML/set_open3d_ml_root.sh'
inference = True
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
remote_weights_path = os.environ['modal.state.slyFile']
device = 'cuda'


root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

train_source_path = os.path.join(root_source_path, "supervisely/train/src")
sly.logger.info(f"Train source directory: {train_source_path}")
sys.path.append(train_source_path)

model = None
meta: sly.ProjectMeta = None
