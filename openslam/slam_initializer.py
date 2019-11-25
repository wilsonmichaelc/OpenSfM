from opensfm import reconstruction
from opensfm import matching
from opensfm import types

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
        im1, im2 = self.ref_frame, frame
        print(im1, im2)
        p1, f1, c1 = data.load_features(im1)
        p2, f2, c2 = data.load_features(im2)
        threshold = data.config['five_point_algo_threshold']
        cameras = data.load_camera_models()
        camera = next(iter(cameras.values()))
        success, matches = self.slam_matcher.match(data, im1, im2, camera)
        if not success:
            return None, None, None #, [], []
        print(p1[:,:3])
        print(len(matches))
        print(matches[im2])
        matches = matches[im2]
        p1 = p1[matches[:, 0], :]
        p2 = p2[matches[:, 1], :]
        f1, f2 = f1[matches[:, 0], :], f2[matches[:, 1], :]
        c1, c2 = c1[matches[:, 0], :], c2[matches[:, 1], :]
        print(p1.shape, p2.shape)
        print(p1)
        threshold = 4 * data.config['five_point_algo_threshold']
        args = []
        args.append((im1, im2, p1[:, 0:2], p2[:, 0:2],
                     camera, camera, threshold))
        # args.append((im1, im2, p1[:,0:2], p2[:,0:2], self.camera_object, self.camera_object, threshold))
        # im1, im2, p1, p2, camera1, camera2, threshold = args
        # print("self.camera_object.pixel_bearing_many(p1)", self.camera_object.pixel_bearing_many(p1))

        i1, i2, r = reconstruction._compute_pair_reconstructability(args[0])
        print("i1:", i1, " i2: ", i2)
        print("r:", r)
        if r == 0:
            return None, None, None
            #return False, [], []

        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(self.ref_frame), bipartite=0)
        tracks_graph.add_node(str(frame), bipartite=0)
        # only add the matches
        for i in range(0, len(f1)):
            feature_id = i
            x, y, s = p1[feature_id, :-1]
            r, g, b = c1[feature_id, :]
            tracks_graph.add_node(str(i), bipartite=1)
            tracks_graph.add_edge(str(self.ref_frame),
                                  str(i),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(feature_id),
                                  feature_color=(float(r), float(g), float(b)))
            x, y, s = p2[feature_id, :-1]
            r, g, b = c2[feature_id, :]
            tracks_graph.add_edge(str(frame),
                                  str(i),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(feature_id),
                                  feature_color=(float(r), float(g), float(b)))

        rec_report = {}
        reconstruction_init, graph_inliers, rec_report['bootstrap'] = \
            reconstruction.bootstrap_reconstruction(data, tracks_graph,
                                                    self.ref_frame, frame, p1,
                                                    p2)

        print("reconstruction", reconstruction_init)
        return reconstruction_init, graph_inliers, matches

    def initialize_opensfm2(self, data, frame):
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
        success, matches = self.slam_matcher.match(data, im1, im2, camera)
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
        return success, matches

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
