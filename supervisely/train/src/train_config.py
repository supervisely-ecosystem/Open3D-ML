import os
import yaml
import sly_globals as g


def save_config(cfg, config_save_path):
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)



def classes_ranges_and_sizes_pointpillars(cfg, state):
    classes = state['selectedTags']
    default_iou = cfg['model']['head']['iou_thr']
    ranges_dict = g.api.app.get_field(g.task_id, "data.ranges")
    sizes_dict = g.api.app.get_field(g.task_id, "data.sizes")

    new_iou = []
    new_ranges = []
    new_sizes = []

    for cl in classes:
        new_iou.append(default_iou[0])
        new_ranges.append(ranges_dict[cl])
        new_sizes.append(sizes_dict[cl])

    cfg['model']['classes'] = classes
    cfg['model']['head']['ranges'] = new_ranges
    cfg['model']['head']['sizes'] = new_sizes
    cfg['model']['head']['iou_thr'] = new_iou
    return cfg

def generate_config(state):
    with open(os.path.join(g.root_source_dir, state["modelConfigExample"]), 'r') as f:
        cfg = yaml.safe_load(f)

    if state['selectedModel'] == "PointPillars":
        cfg = classes_ranges_and_sizes_pointpillars(cfg, state)
    else:
        cfg['model']['classes'] = state['selectedTags']

    cfg['dataset']['project_path'] = g.project_dir
    cfg['dataset']['steps_per_epoch_train'] = state["steps_per_epoch_train"]
    cfg['dataset']['val_split'] = g.api.app.get_field(g.task_id, "data.valImagesCount")

    cfg['model']['ckpt_path'] = state['localWeightsPath']

    # cfg['model']['augment']['PointShuffle'] = state['pointShuffle']
    # cfg['model']['augment']['ObjectNoise'] = state['objectNoise'] # TODO: make augs available
    # cfg['model']['augment']['RangeFilter'] = state['rangeFilter']

    cfg['pipeline']['batch_size'] = state["batchSizeTrain"]
    cfg['pipeline']['val_batch_size'] = state["batchSizeVal"]
    cfg['pipeline']['save_ckpt_freq'] = state["checkpointInterval"]
    cfg['pipeline']['max_epoch'] = state["epochs"] - 1
    cfg['pipeline']['grad_clip_norm'] = state["gradClipNorm"]

    cfg['pipeline']['optimizer']['lr'] = state["lr"]
    cfg['pipeline']['optimizer']['weight_decay'] = state["weightDecay"]

    return cfg


def save_from_state(state):
    with open(state["trainConfigPath"], 'w') as f:
        f.write(state["trainConfigLines"])
