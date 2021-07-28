import functools
import os

import supervisely_lib as sly

import globals as g
import nn_utils


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("get_output_classes_and_tags")
@sly.timeit
@send_error_data
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "Open3D Detection Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("inference_pointcloud_id")
@sly.timeit
@send_error_data
def inference_pointcloud_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    local_pointcloud_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ".pcd")
    api.pointcloud.download_path(state["pointcloud_id"], local_pointcloud_path)
    #
    results = nn_utils.inference_model(g.model, local_pointcloud_path)
    # sly.fs.silent_remove(local_pointclod_path)


    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"results": results})

def debug_inference():
    results = nn_utils.inference_model(g.model,  "/data/test_cloud.pcd")
    print(results)

def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()

    g.my_app.run()

if __name__ == "__main__":
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()

    #sly.main_wrapper("main", main)
    debug_inference()