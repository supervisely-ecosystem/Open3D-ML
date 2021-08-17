import os
import supervisely_lib as sly

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
remote_weights_path = os.environ['modal.state.slyFile']
device = os.environ['modal.state.device']

model = None
meta: sly.ProjectMeta = None
