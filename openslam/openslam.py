import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from slam_initializer import SlamInitializer
from slam_matcher import SlamMatcher
from slam_mapper import SlamMapper
from slam_tracker import SlamTracker
from slam_frame import Frame
from opensfm import dataset
from opensfm import features
from opensfm import reconstruction
from opensfm import feature_loader
# from initializer import slam_initializer
import numpy as np
from opensfm import feature_loading
import networkx as nx
from opensfm import log
import logging

log.setup()
logger = logging.getLogger(__name__)


class SlamSystem(object):

    def __init__(self, args):
        print("Init slam system", args)
        self.data = dataset.DataSet(args.dataset)
        cameras = self.data.load_camera_models()
        print("cameras.items()", cameras.items())
        print("cameras.values()", cameras.values())
        self.config = self.data.config
        self.camera = next(iter(cameras.items()))
        self.camera_object = next(iter(cameras.values()))
        self.system_initialized = False
        self.system_lost = True
        self.slam_matcher = SlamMatcher(self.config)
        self.initializer = SlamInitializer(self.config, self.slam_matcher)
        self.initializer.matcher = self.slam_matcher
        self.tracked_frames = 0
        self.reconstruction_init = None
        self.image_list = sorted(self.data.image_list)
        self.slam_mapper = SlamMapper(self.data, self.config)
        self.slam_tracker = SlamTracker(self.data, self.config)

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    # def run(self,args)
    #     print("Init slam system")
    #     data = dataset.DataSet(args.dataset)
    #     camera = data.load_camera_models[0]
    #     print("camera")
    #     self.camera = camera
    #     self.system_initialized = False
    #     self.system_lost = True

    def init_slam_system(self, data, frame):
        """Find the initial depth estimates for the slam map"""
        print("init_slam_system: ", frame)
        if self.tracked_frames == 0:
            self.initializer.set_initial_frame(data, frame)
            self.tracked_frames += 1
            matches = []
            return False, None, None, None
        else:
            print("Trying initialize")
            reconstruction_init, graph_inliers, matches = self.initializer.initialize(data, frame)
            print("Tried to initialize ", reconstruction_init)
            self.system_initialized = (reconstruction_init is not None)
            
            print("init: {}, matches {} ".format(self.system_initialized, matches))
            if (self.system_initialized):
                print("System initialized!", len(matches))
                # print("Returning", self.system_initialized, len(matches), len(matches[1][frame]))
                # print(graph_inliers[str(frame)])
            print(self.tracked_frames, self.system_initialized, matches)
            return self.system_initialized, reconstruction_init, graph_inliers, matches

    def track_next_frame(self, data, frame: str):
        """Estimates the pose for the next frame"""
        if not self.system_initialized:
            self.system_initialized, self.reconstruction_init, graph_inliers, matches = self.init_slam_system(data, frame)
            if self.system_initialized:
                self.tracked_frames += 1
                self.slam_mapper.reconstruction = self.reconstruction_init
                # init successful, create new kf
                kf = Frame(frame)
                print(matches[:, 1].max())
                print(graph_inliers[str(frame)])
                print(len(graph_inliers[str(frame)].keys()))

                # Take all points as landmarks
                kf.set_visible_landmarks(self.reconstruction_init.points, graph_inliers[str(frame)])
                self.slam_mapper.graph = graph_inliers
                # , matches[:, 1])
                self.slam_mapper.set_last_keyframe(kf)
            print("Init: ", self.system_initialized)
            return self.system_initialized
        else:
            print("success")
            print("self.reconstruction_init: ",
                  len(self.reconstruction_init.points))
            self.slam_tracker.track(self.slam_mapper, frame,
                                    self.config, self.camera, data)

            return True
        return False

            # im1, im2 = self.initializer.ref_frame.im_name, frame.im_name
            # print(im1, im2)
            # p1, f1, _ = data.load_features(im1)
            # # p1v, f1v, _ = feature_loader.instance.load_points_features_colors(
            # #    data, im1, masked=True)
            # # print("p1:",p1, "p1v:", p1v)
            # # print("f1:",f1, "f1v:", f1v)
            # # print(p1.shape,p1v.shape,f1.shape,f1v.shape)
            # p2, f2, _ = data.load_features(im2)
            # #feature_loader.instance.load_points_features_colors(
            # #    data, im2, masked=True)
            # threshold = data.config['five_point_algo_threshold']
            # if not success:
            #     self.tracked_frames += 1
            #     return False
            # print(p1[:,:3])
            # print(len(matches))
            # print(matches[frame.im_name])
            # matches = matches[frame.im_name]
            # # print(len(matches))
            # # print(len(p1),len(p2))
            # # print(p1.shape, p2.shape)
            # # print(p1)
            # p1 = p1[matches[:,0],:]
            # p2 = p2[matches[:,1],:]
            # # print(len(p1),len(p2))
            # print(p1.shape, p2.shape)
            # print(p1)
            # threshold = 4 * data.config['five_point_algo_threshold']
            # args = []
            # args.append((im1, im2, p1[:,0:2], p2[:,0:2], self.camera_object, self.camera_object, threshold))
            # # im1, im2, p1, p2, camera1, camera2, threshold = args
            # # print("self.camera_object.pixel_bearing_many(p1)", self.camera_object.pixel_bearing_many(p1))
            
            # i1, i2, r = reconstruction._compute_pair_reconstructability(args[0])
            # print("i1:", i1, " i2: ", i2)
            # print("r:", r)
            # # stop
            # # print(self.camera)
            # # print(self.camera[1].pixel_bearing_many(p1))
            # R, t, inliers,_ = \
            #     reconstruction.two_view_reconstruction_general(p1, p2, self.camera[1], self.camera[1], threshold)
            # if len(inliers) <= 5:
            #     report['decision'] = "Could not find initial motion"
            #     logger.info(report['decision'])
            #     return False
            #     # return None, None, report
            # # print("here",p1)
            # # print(len(p1),len(p1(matches[:,0])))
            # print("R: ", R, " t: ", t)
            # print("inliers: ", inliers, len(inliers))



            # if success:
                # print("success")
                # stop
                #initialized
                # self.slam_mapper.add_new_tracks(self.initializer.ref_frame.im_name,
                                                # frame.im_name, matches)
                # slam_matcher.match(self.initializer)
                # matching.match
                # system is initialized now
                # add first frame and current frame to the graph!        
                # self.global_graph.add_node(str(self.initializer.ref_frame.im_name), bipartite=0)
                # self.global_graph.add_node(str(frame.im_name), bipartite=0)
                #now add the matches between both -> where do I get the matches from?
                #for track_id 
                # tracks_graph.add_node(str(track_id), bipartite=1)
                # tracks_graph.add_edge(str(image),
                #                     str(track_id),
                #                     feature=(float(x), float(y)),
                #                     feature_scale=float(s),
                #                     feature_id=int(featureid),
                #                     feature_color=(float(r), float(g), float(b)))
                
                # TODO: Use _good_track....
        self.tracked_frames += 1
        return not self.system_lost

    def local_optimization(self):
        return True

    def global_optimization(self):
        return True

    def reset(self):
        """Reset the system"""
        return True

    def relocalize(self, frame):
        """Relocalize against the SLAM map either after tracking loss or
           alternatively if the region is sufficiently explored
        """
        return True

    def save_slam_reconstruction(self):
        """Saves the slam recontsruction in the OpenSfM format"""
        return True

    def save_slam_trajectory(self, kf_only=False):
        """Saves the trajectory file of all frames or only of KFs"""
        return True

    def detect(self, data, frame):
        image = self.image_list[self.tracked_frames]
        p_sorted, f_sorted, c_sorted = self.feature_loader.load_points_features_colors(data,image)
        if p_sorted is None or f_sorted is None or c_sorted is None:
            p_unmasked, f_unmasked, c_unmasked = features.extract_features(
                data.load_image(image), data.config)

            fmask = self.data.load_features_mask(image, p_unmasked)

            p_unsorted = p_unmasked[fmask]
            f_unsorted = f_unmasked[fmask]
            c_unsorted = c_unmasked[fmask]

            if len(p_unsorted) == 0:
                logger.warning('No features found in image {}'.format(image))
                return

            size = p_unsorted[:, 2]
            order = np.argsort(size)
            p_sorted = p_unsorted[order, :]
            f_sorted = f_unsorted[order, :]
            c_sorted = c_unsorted[order, :]
            data.save_features(image, p_sorted, f_sorted, c_sorted)

