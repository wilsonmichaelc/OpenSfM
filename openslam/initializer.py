from opensfm import reconstruction


class slam_initializer(object):

    def __init__(self):
        print("initializer")
        self.init_type = "ICCV"

    def set_initial_frame(self, frame):
        """Sets the first frame"""

    def initialize_opensfm(self, frame):
        """Tries to initialize with respect to the frist frame"""
        threshold = data.config['five_point_algo_threshold']
        min_inliers = data.config['five_point_algo_min_inliers']
        #p1, p2 features, camera1 == camera2
        R, t, inliers, report['two_view_reconstruction'] = \
            reconstruction.two_view_reconstruction_general(p1, p2, camera1, camera2, threshold)

    def initialize_openvslam(self, frame):
        """Initialize similar to ORB-SLAM and Openvslam"""
        print("initialize_openvslam")

    def initialize_iccv(self, frame):
        """Initialize similar Elaborate Monocular Point and Line SLAM 
            With Robust Initialization
        """
        print("initialize_openvslam")

    def initialize(self, frame):
        if self.init_type == "ICCV":
            return self.initialize_iccv(frame)
        if self.init_type == "OpenSfM":
            return self.initialize_opensfm(frame)
        return self.initialize_openvslam(frame)
