from collections import defaultdict
import supervisely_lib as sly

import input_project
import splits
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

tag2images = None
tag2urls = None
pointclouds_without_figures = []
disabled_tags = []
tags_count = {}

progress_index = 3
_preview_height = 120
_max_examples_count = 12

_ignore_tags = ["train", "val"]
_allowed_tag_types = [sly.geometry.cuboid_3d.Cuboid3d]

image_slider_options = {
    "selectable": False,
    "height": f"{_preview_height}px"
}

selected_tags = None


def init(data, state):
    data["tagsBalance"] = None
    state["selectedTags"] = []
    state["tagsInProgress"] = False
    data["tagsBalanceOptions"] = {
        "selectable": True,
        "collapsable": True,
        "clickableName": False,
        "clickableSegment": False,
        "maxHeight": "400px"
    }
    data["imageSliderOptions"] = image_slider_options
    data["done3"] = False
    data["skippedTags"] = []
    state["collapsed3"] = True
    state["disabled3"] = True
    init_progress(progress_index, data)


def restart(data, state):
    data["done3"] = False


def init_cache(progress_cb):
    global tags_count, pointclouds_without_figures
    project_fs = sly.PointcloudProject.read_single(g.project_dir)
    tag_names = []
    for dataset_fs in project_fs:
        for item_name in dataset_fs:
            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            ann_json = sly.io.json.load_json_file(ann_path)
            ann = sly.PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            if len(ann.figures) == 0:
                 pointclouds_without_figures.append(item_name)
            else:
                for fig in ann.figures:
                    tag_name = fig.parent_object.obj_class.name
                    tag_names.append(tag_name)

    for tag_name in tag_names:
        tags_count[tag_name] = tag_names.count(tag_name)

    progress_cb(1)


@g.my_app.callback("show_tags")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def show_tags(api: sly.Api, task_id, context, state, app_logger):
    global tags_count, pointclouds_without_figures

    progress = get_progress_cb(progress_index, "Calculate stats", g.project_info.items_count)

    init_cache(progress)


    disabled_tags = []
    _working_tags = tags_count.keys()
    tags_balance_rows = []

    max_count = -1
    for tag_meta in g.project_meta.tag_metas:
        if tag_meta.name not in _working_tags:
            # tags with 0 images will be ignored automatically
            disabled_tags.append({
                "name": tag_meta.name,
                "color": sly.color.rgb2hex(tag_meta.color),
                "reason": "0 pointclouds with this tag"
            })


            tags_balance_rows.append({
                "name": tag_meta.name,
                "total": tags_count[tag_meta.name],
                "disabled": True})
        else:

            tags_balance_rows.append({
                "name": tag_meta.name,
                "total": tags_count[tag_meta.name],
                "disabled": False if tags_count > 0 else True})

        max_count = max(max_count, tags_count[tag_meta.name])

    rows_sorted = sorted(tags_balance_rows, key=lambda k: k["total"], reverse=True)
    tags_balance = {
        "maxValue": max_count,
        "rows": rows_sorted
    }

    subsample_urls = {tag_name: count for tag_name, count in tags_count.items()}

    reset_progress(progress_index)

    fields = [
        {"field": "state.tagsInProgress", "payload": False},
        {"field": "data.tagsBalance", "payload": tags_balance},
        {"field": "data.tag2urls", "payload": subsample_urls},
        {"field": "data.skippedTags", "payload": disabled_tags}
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("use_tags")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_tags(api: sly.Api, task_id, context, state, app_logger):
    global selected_tags
    selected_tags = state["selectedTags"]

    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},
        {"field": "state.activeStep", "payload": 4},
    ]
    g.api.app.set_fields(g.task_id, fields)