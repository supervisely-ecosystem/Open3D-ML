import os
import yaml
import sly_globals as g


def save_config(cfg, config_save_path):
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)


def classes_ranges_and_sizes_pointpillars(cfg, state):
    classes = state['selectedTags']
    default_classes = cfg['model']['classes']
    default_ranges = cfg['model']['head']['ranges']
    default_sizes = cfg['model']['head']['sizes']
    default_iou = cfg['model']['head']['iou_thr']

    if classes == default_classes:
        return cfg

    new_ranges = []
    new_sizes = []
    new_iou = []

    for cl in classes:
        try:
            ind = default_classes.index(cl)
            new_ranges.append(default_ranges[ind])
            new_sizes.append(default_sizes[ind])
            new_iou.append(default_iou[ind])
        except ValueError:
            new_ranges.append(default_ranges[0]) # TODO: calc it!
            new_sizes.append(default_sizes[0])
            new_iou.append(default_iou[0])

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

    cfg['dataset']['dataset_path'] = g.project_dir + "_kitti"
    cfg['dataset']['steps_per_epoch_train'] = state["steps_per_epoch_train"]
    cfg['dataset']['val_split'] = g.api.app.get_field(g.task_id, "data.trainImagesCount") + 1

    cfg['model']['ckpt_path'] = state['localWeightsPath']

    cfg['model']['augment']['PointShuffle'] = state['pointShuffle']
    cfg['model']['augment']['ObjectNoise'] = state['objectNoise']
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
