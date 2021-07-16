from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection, \
    convert_range_image_to_point_cloud

from scene.base_scene import BaseScene
import numpy as np
import cv2


class WaymoScene(BaseScene):
    def __init__(self, input_data):
        self.data = input_data
        self.class_map = ['Unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
        self.range_images, self.camera_projections, self.range_image_top_pose = \
            parse_range_image_and_camera_projection(self.data)

    @property
    def images(self):
        def imdecode(x): return cv2.imdecode(np.frombuffer(x.image, np.uint8), cv2.IMREAD_COLOR)
        return [imdecode(x) for x in self.data.images]

    @property
    def pointcloud(self):
        points_0, _ = convert_range_image_to_point_cloud(self.data,
                                                         self.range_images,
                                                         self.camera_projections,
                                                         self.range_image_top_pose,
                                                         ri_index=0)

        points_1, _ = convert_range_image_to_point_cloud(self.data,
                                                         self.range_images,
                                                         self.camera_projections,
                                                         self.range_image_top_pose,
                                                         ri_index=1)

        points0 = np.concatenate(points_0, axis=0)
        points_1 = np.concatenate(points_1, axis=0)
        points = np.concatenate([points0, points_1], axis=0)
        return points.astype(np.float32)

    @property
    def bboxes_3d(self):
        bboxes = []
        for obj in self.data.laser_labels:
            h = obj.box.height  # up/down
            w = obj.box.width  # left/right
            l = obj.box.length  # front/back

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z
            yaw = obj.box.heading - np.pi / 2
            bboxes.append([x, y, z, w, l, h, yaw])

        return bboxes

    @property
    def labels(self):
        return [self.class_map[obj.type] for obj in self.data.laser_labels]

    @property
    def name(self):
        # lidar sweep time in microseconds, used as the name of the point cloud
        return str(self.data.timestamp_micros)  # TODO: something that guarantee unique values
