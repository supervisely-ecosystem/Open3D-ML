from abc import ABC, abstractmethod


class BaseScene(ABC):
    """
        Scene is definition of middle format between sly and other datasets / inference modules
        Contains: pointcloud array and labels
    """

    @property
    @abstractmethod
    def images(self):
        """
        :return: list of RGB images in format like cv2.imread does
        """
        pass

    @property
    @abstractmethod
    def pointcloud(self):
        """
        :return: pointcloud in basic format N x 3, i.e.
            [[x,y,z],
             [x,y,z]
              ...
             [x,y,z]]

        Where x,y,z are python floats
        """
        pass

    @property
    @abstractmethod
    def bboxes_3d(self):
        """
        x,y,z - center_point
        w,l,h - width, lenght and height of bbox
        yaw - [-pi;pi]
        :return: list of 3d bboxes [[x, y, z, w, l, h, yaw], [x, y, z, w, l, h, yaw]]  in same order with labels
        """
        pass


    @property
    @abstractmethod
    def labels(self):
        """
        :return: list of labels like ['Car', 'Pedestrian']. Order respects bboxes_3d
        """
        pass

    @property
    def name(self):
        """
            str name of pointcloud
        """
        pass