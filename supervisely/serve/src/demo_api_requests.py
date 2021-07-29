import json
import supervisely_lib as sly


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 7319

    # get information about model
    print('\n')
    info = api.task.send_request(task_id, "get_session_info", data={})
    print("Information about deployed model:")
    print(json.dumps(info, indent=4))

    # get model output tags
    print('\n')
    meta_json = api.task.send_request(task_id, "get_output_classes_and_tags", data={})
    model_meta = sly.ProjectMeta.from_json(meta_json)
    print("Model predicts following tags:")
    print(model_meta)

    # get model custom inference settings
    print('\n')
    r = api.task.send_request(task_id, "get_custom_inference_settings", data={})
    print("Model has following custom settings:")
    print(json.dumps(r, indent=4))

    print('\n')
    r = api.task.send_request(task_id, "inference_pointcloud_id", data={"pointcloud_id": 938281})
    print("Model prediction for single pointcloud:")
    print(json.dumps(r, indent=4))

    print('\n')
    r = api.task.send_request(task_id, "inference_pointcloud_ids", data={"pointcloud_ids": [938281, 938281],
                                                                         "threshold":0.7})
    print(json.dumps(r, indent=4))

if __name__ == "__main__":
    main()