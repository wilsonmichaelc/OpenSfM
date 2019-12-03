import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from slam_initializer import SlamInitializer
from slam_matcher import SlamMatcher
from slam_mapper import SlamMapper
from slam_tracker import SlamTracker
from slam_types import Frame
from slam_types import Keyframe
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
        if self.initializer.init_frame is None:
            self.initializer.set_initial_frame(data, frame)
            # self.tracked_frames += 1
            matches = []
            return False #, None, None, None
        else:
            print("Trying initialize")
            reconstruction_init, graph_inliers, matches = self.initializer.initialize(data, frame)
            print("Tried to initialize ", reconstruction_init)
            self.system_initialized = (reconstruction_init is not None)
            
            # print("init: {}, matches {} ".format(self.system_initialized, matches))
            if (self.system_initialized):
                print("System initialized!", len(matches))
                # print(reconstruction_init.shots)
                # print(reconstruction_init.points)
                print(graph_inliers.nodes())
                # print(graph_inliers['00000.png',])
                self.slam_mapper.create_init_map(graph_inliers,
                                                 reconstruction_init,
                                                 self.initializer.init_frame,
                                                 frame)
                # exit()
                # #let's create two keyframes and add them to the mapper
                # init_keyfrm = Keyframe(self.initializer.init_frame)
                # ref_keyfrm = Keyframe(frame)
                # self.slam_mapper.
            # print(self.tracked_frames, self.system_initialized, matches)
            return self.system_initialized #, reconstruction_init, graph_inliers, matches

    def track_next_frame(self, data, frame: Frame):
        """Estimates the pose for the next frame"""
        if not self.system_initialized:
            self.system_initialized = self.init_slam_system(data, frame)
            # self.reconstruction_init, graph_inliers, matches = self.init_slam_system(data, frame)
            if self.system_initialized:
                # frame.id = self.tracked_frames
                self.tracked_frames += 1
                # self.slam_mapper.reconstruction = self.reconstruction_init
                # init successful, create new kf
                # kf = Frame(frame.im_name)
                # Take all points as landmarks
                # kf.set_visible_landmarks(self.reconstruction_init.points, graph_inliers[str(frame.im_name)])
                # self.slam_mapper.graph = graph_inliers
                # , matches[:, 1])
                self.slam_mapper.update_local_map()
                # self.slam_mapper.set_curr_kf(kf)
            print("Init: ", self.system_initialized)
            return self.system_initialized
        else:
            print("success")
            print("self.reconstruction_init: ",
                  len(self.slam_mapper.reconstruction.points))
            
            # Maybe remove most of the slam_mapper stuff
            # to tracking
            self.slam_mapper.apply_landmark_replace()
            # self.slam_mapper.update_last_keyframe()
            # frame.
            pose = self.slam_tracker.track(self.slam_mapper, frame,
                                           self.config, self.camera, data)
            if pose is not None:
                self.slam_mapper.update_local_map()
                pose = self.slam_mapper.track_with_local_map()
                if pose is not None:
                    self.tracked_frames += 1
                    # Store the map
                    self.slam_mapper.add_frame_to_reconstruction(frame, pose, self.camera, data)
                    self.slam_mapper.paint_reconstruction(data)
                    self.slam_mapper.save_reconstruction(data, frame)
                    new_kf = self.slam_mapper.new_kf_needed(1000, frame)
                    print("New kf needed: ", new_kf)
            return True
        # return False

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

