from opensfm import reconstruction
from opensfm import matching
import logging
import networkx as nx
logger = logging.getLogger(__name__)



class SlamInitializer(object):

    def __init__(self, config, slam_matcher):
        print("initializer")
        self.init_type = "OpenSfM"
        self.ref_frame = None
        self.slam_matcher = slam_matcher

    def set_initial_frame(self, data, frame):
        """Sets the first frame"""
        self.ref_frame = frame

    def initialize_opensfm(self, data, frame):
        """Tries to initialize with respect to the frist frame"""
        #TODO: Super slow at the moment

        logger.info("Starting reconstruction with {} and {}".
                        format(self.ref_frame.im_name, frame.im_name))
        print("Starting reconstruction with {} and {}".
                        format(self.ref_frame.im_name, frame.im_name))
        im1, im2 = self.ref_frame.im_name, frame.im_name
        camera = next(iter(data.load_camera_models().values()))
        # print("camera: ", camera)
        #PerspectiveCamera('v2 unknown unknown 1024 768 perspective 0.9722', 'perspective', 1024, 768, 0.9722222222222222, 0.0, 0.0, 0.9722222222222222, 0.0, 0.0)
        # print("camera: ", camera[data.ex[im1]['camera']])
        # print("test: ", im1, im2)
        # matches = matching.match(im1, im2, camera, camera, data)
        matches = self.slam_matcher.match(data, im1, im2, camera)
        print("Matches: ", matches)
        # threshold = data.config['five_point_algo_threshold']
        # min_inliers = data.config['five_point_algo_min_inliers']
        # p1, f1, c1 = self.ref_frame.load_points_features_clr(data)
        # p2, f2, c2 = frame.load_points_features_clr(data)
        # # print("p1 before ", len(p1), matches)
        # print("p1 before ", p1.shape, f1.shape, c1.shape)
        # print("p2 before ", p2.shape, f2.shape, c2.shape)
        # print("matches: ", matches)
        # if len(matches) < min_inliers:
        #     return False
        # p1 = p1[matches[:, 0], 0:2]
        # p2 = p2[matches[:, 1], 0:2]
        # print("p1 after ", len(p1))
        # print(p1, p2)
        # report = {
        #     'image_pair': (im1, im2),
        #     'common_tracks': len(p1),
        # }
        # graph = nx.Graph()
        # graph.add_node(str(im1), bipartite=0)
        # graph.add_node(str(im2), bipartite=0)
        # graph.add_node('0', bipartite=1)
        # # graph.add_node('1', bipartite=1)
        # graph.add_edge(str(im1),
        #                 '0',
        #                 feature=(float(x), float(y)),
        #                 feature_scale=float(s),
        #                 feature_id=int(featureid),
        #                 feature_color=(float(r), float(g), float(b)))
        
        # graph.add_edge(str(im2),
        #                 '0',
        #                 feature=(float(x), float(y)),
        #                 feature_scale=float(s),
        #                 feature_id=int(featureid),
        #                 feature_color=(float(r), float(g), float(b)))
        # reconstruction.bootstrap_reconstruction(data,graph,im1,im2,p1,p2)

        # R, t, inliers, report['two_view_reconstruction'] = reconstruction.two_view_reconstruction_general(p1, p2, camera, camera, threshold)
        # logger.info("Two-view reconstruction inliers: {} / {}".format(
        #     len(inliers), len(p1)))

        # if len(inliers) <= 5:
        #     report['decision'] = "Could not find initial motion"
        #     logger.info(report['decision'])
        #     return False
        return matches

    def initialize_openvslam(self, data, frame):
        """Initialize similar to ORB-SLAM and Openvslam"""
        print("initialize_openvslam")

    def initialize_iccv(self, data, frame):
        """Initialize similar Elaborate Monocular Point and Line SLAM 
            With Robust Initialization
        """
        print("initialize_openvslam")

    def initialize(self, data, frame):
        if self.init_type == "ICCV":
            return self.initialize_iccv(data, frame)
        if self.init_type == "OpenSfM":
            return self.initialize_opensfm(data, frame)
        return self.initialize_openvslam(data, frame)
