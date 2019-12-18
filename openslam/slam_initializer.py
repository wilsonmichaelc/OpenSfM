from opensfm import reconstruction
from opensfm import matching
from opensfm import types
from opensfm import feature_loader
import slam_debug
import numpy as np
import logging
import networkx as nx
logger = logging.getLogger(__name__)


class SlamInitializer(object):

    def __init__(self, config, slam_matcher):
        print("initializer")
        self.init_type = "OpenSfM"
        self.init_frame = None
        self.slam_matcher = slam_matcher

    def set_initial_frame(self, data, frame):
        """Sets the first frame"""
        self.init_frame = frame

    def initialize_opensfm(self, data, frame):
        # im1, im2 = self.init_frame.im_name, frame.im_name
        im2, im1 = self.init_frame.im_name, frame.im_name
        print(im1, im2)
        p1, f1, c1 = feature_loader.instance.\
            load_points_features_colors(data, im1, masked=True)
        p2, f2, c2 = feature_loader.instance.\
            load_points_features_colors(data, im2, masked=True)
        threshold = data.config['five_point_algo_threshold']
        cameras = data.load_camera_models()
        camera = next(iter(cameras.values()))
        success, matches = self.slam_matcher.match(data, im1, im2, camera)
        print("cameras", cameras)
        print("camera", camera, camera.k1, camera.k2,
              camera.focal)
        if not success:
            return None, None, None
        matches = matches[im2]
        print("len(p1): ", len(p1), " len(p2): ", len(p2))
        print("p1: ", p1.shape, " p2: ", p2.shape)
        p1 = p1[matches[:, 0], :]
        p2 = p2[matches[:, 1], :]
        f1, f2 = f1[matches[:, 0], :], f2[matches[:, 1], :]
        c1, c2 = c1[matches[:, 0], :], c2[matches[:, 1], :]

        threshold = 4 * data.config['five_point_algo_threshold']
        args = []
        args.append((im1, im2, p1[:, 0:2], p2[:, 0:2],
                     camera, camera, threshold))
        i1, i2, r = reconstruction._compute_pair_reconstructability(args[0])
        if r == 0:
            return None, None, None

        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(im1), bipartite=0)
        tracks_graph.add_node(str(im2), bipartite=0)
        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            x, y, s = p1[track_id, 0:3]
            r, g, b = c1[track_id, :]
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(im1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))
            x, y, s = p2[track_id, 0:3]
            r, g, b = c2[track_id, :]
            tracks_graph.add_edge(str(im2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))
        rec_report = {}
        print("p1 ", p1.shape, " p2 ", p2.shape)
        reconstruction_init, graph_inliers, rec_report['bootstrap'] = \
            reconstruction.bootstrap_reconstruction(data, tracks_graph,
                                                    im1, im2, p1[:, 0:2], p2[:, 0:2])
                                                
        print("Created init rec from {}<->{} with {} points from {} matches"
              .format(im1, im2, len(reconstruction_init.points), len(matches)))
        # seen_landmarks = graph_inliers[im1]
        # print("im1: ", im1, " im2 ", im2)
        # in_graph = {}
        # for lm_id in seen_landmarks:
        #     e = graph_inliers.get_edge_data(im1, lm_id)
        #     print("init opensfm: e(", im1, ",", lm_id, "): ", e)
        #     e2 = graph_inliers.get_edge_data(im2, lm_id)
        #     print("init opensfm: e2(", im2, ",", lm_id, "): ", e2)
        #     if e['feature_id'] in in_graph:
        #         print("Already in there! init", e['feature_id'], "lm_id: ", lm_id)
        #         exit()
        #     in_graph[e['feature_id']] = lm_id
        
        return reconstruction_init, graph_inliers, matches

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
