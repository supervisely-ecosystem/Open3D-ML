import os
import supervisely_lib as sly
import sly_globals as g
import yaml



def save_config(cfg, config_save_path):



    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)

def generate_config(state):
    with open(os.path.join(g.root_source_dir, state["modelConfigExample"]), 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['dataset']['dataset_path'] = g.project_dir + "_kitti"
    cfg['dataset']['steps_per_epoch_train'] = state["steps_per_epoch_train"]
    cfg['dataset']['val_split'] = g.api.app.get_field(g.task_id, "data.trainImagesCount")

    cfg['model']['ckpt_path'] = state['localWeightsPath']
    cfg['model']['augment']['PointShuffle'] = state['pointShuffle']
    cfg['model']['augment']['ObjectNoise'] = state['objectNoise']
    #cfg['model']['augment']['RangeFilter'] = state['rangeFilter']

    cfg['pipeline']['batch_size'] = state["batchSizeTrain"]
    cfg['pipeline']['val_batch_size'] = state["batchSizeVal"]
    cfg['pipeline']['save_ckpt_freq'] = state["checkpointInterval"]
    cfg['pipeline']['max_epoch'] = state["epochs"]
    cfg['pipeline']['grad_clip_norm'] = state["gradClipNorm"]

    cfg['pipeline']['optimizer']['lr'] = state["lr"]
    cfg['pipeline']['optimizer']['weight_decay'] = state["weightDecay"]

    return cfg

def save_from_state(state):
    with open(state["trainConfigPath"], 'w') as f:
        f.write(state["trainConfigLines"])

#cfg = generate_config("/Open3D-ML/supervisely/src_backup/pointpillars_kitti_sly.yml", None)

#save_config(cfg, "/Open3D-ML/supervisely/src_backup/demo.yml")

    # optimizer = f"optimizer = dict(type='{state['optimizer']}', " \
    #             f"lr={state['lr']}, " \
    #             f"momentum={state['momentum']}, " \
    #             f"weight_decay={state['weightDecay']}" \
    #             f"{', nesterov=True' if (state['nesterov'] is True and state.optimizer == 'SGD') else ''})"
    #
    # grad_clip = f"optimizer_config = dict(grad_clip=None)"
    # if state["gradClipEnabled"] is True:
    #     grad_clip = f"optimizer_config = dict(grad_clip=dict(max_norm={state['maxNorm']}))"
    #
    # lr_updater = ""
    # if state["lrPolicyEnabled"] is True:
    #     py_text = state["lrPolicyPyConfig"]
    #     py_lines = py_text.splitlines()
    #     num_uncommented = 0
    #     for line in py_lines:
    #         res_line = line.strip()
    #         if res_line != "" and res_line[0] != "#":
    #             lr_updater += res_line
    #             num_uncommented += 1
    #     if num_uncommented == 0:
    #         raise ValueError("LR policy is enabled but not defined, please uncomment and modify one of the provided examples")
    #     if num_uncommented > 1:
    #         raise ValueError("several LR policies were uncommented, please keep only one")
    #
    # runner = f"runner = dict(type='EpochBasedRunner', max_epochs={state['epochs']})"
    # if lr_updater == "":
    #     lr_updater = "lr_config = dict(policy='fixed')"
    # py_config = optimizer + os.linesep + \
    #             grad_clip + os.linesep + \
    #             lr_updater + os.linesep + \
    #             runner + os.linesep
    #
    # with open(schedule_config_path, 'w') as f:
    #     f.write(py_config)
    # return schedule_config_path, py_config
    #
    # with open(runtime_config_path, 'w') as f:
    #     f.write(py_config)
    #
    # return runtime_config_path, py_config



