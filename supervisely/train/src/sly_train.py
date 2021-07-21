import supervisely_lib as sly
import sly_globals as g
import ui as ui
#import sly_logger_hook  # to register hook
#import sly_imgaugs  # to register first part of the pipeline


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    g.my_app.compile_template(g.root_source_dir)

    data = {}
    state = {}
    data["taskId"] = g.task_id

    ui.init(data, state)  # init data for UI widgets
    g.my_app.run(data=data, state=state)  # state a dict (buttons) , data: raw data jpegs et.c.


if __name__ == "__main__":
    sly.main_wrapper("main", main)
