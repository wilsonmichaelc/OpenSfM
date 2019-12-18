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
from opensfm import types
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
        self.slam_mapper = SlamMapper(self.data, self.config, self.camera)
        self.slam_tracker = SlamTracker(self.data, self.config)
        self.n_tracked_lms = 0

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def init_slam_system(self, data, frame: Frame):
        """Find the initial depth estimates for the slam map"""
        print("init_slam_system: ", frame)
        if self.initializer.init_frame is None:
            self.initializer.set_initial_frame(data, frame)
            self.system_initialized = False
        else:
            rec_init, graph, matches = \
                self.initializer.initialize(data, frame)
            self.system_initialized = (rec_init is not None)
            if self.system_initialized:
                self.slam_mapper.create_init_map(graph, rec_init,
                                                 self.initializer.init_frame,
                                                 frame)
        return self.system_initialized

    def track_next_frame(self, data, frame: Frame):
        """Estimates the pose for the next frame"""
        if not self.system_initialized:
            self.system_initialized = self.init_slam_system(data, frame)
            if self.system_initialized:
                print("Initialized system with ", frame.im_name)
            else:
                print("Failed to initialize with ", frame.im_name)
            return self.system_initialized
        else:
            print("Tracking: ", frame.frame_id, frame.im_name)
            # Maybe move most of the slam_mapper stuff to tracking
            self.slam_mapper.apply_landmark_replace()
            #TODO: Update last frames' pose!
            pose = self.slam_tracker.track(self.slam_mapper, frame,
                                           self.config, self.camera, data)
            
            print("pose after track for ", frame.im_name, ": ",
                  pose.rotation, pose.translation)
            if pose is not None:
                frame.world_pose = pose
                self.slam_mapper.update_local_map(frame)
                print("After update_local_map")
                pose: types.Pose = self.slam_mapper.track_with_local_map(frame, self.slam_tracker)
                print("pose after track_with_local_map: ",
                      pose.rotation, pose.translation)
                if pose is not None:
                    self.tracked_frames += 1
                    # Store the map
                    # self.slam_mapper.\
                        # add_frame_to_reconstruction(frame.im_name, pose, self.camera, data)
                    
                    # TODO: Check that
                    self.slam_mapper.set_last_frame(frame)
                    if self.slam_mapper.new_kf_needed(frame):
                        new_kf = Keyframe(frame, data, self.slam_mapper.n_keyframes)
                        self.slam_mapper.add_keyframe(new_kf)
                        # self.slam_mapper.curr_kf = new_kf
                        self.slam_mapper.mapping_with_new_keyframe(new_kf)
                        self.slam_mapper.paint_reconstruction(data)
                        self.slam_mapper.save_reconstruction(data, frame.im_name)
                        self.slam_mapper.local_bundle_adjustment()
                        self.slam_mapper.paint_reconstruction(data)
                        self.slam_mapper.save_reconstruction(data, frame.im_name+"aft")
                        print("New kf needed: ", new_kf)
                    else:
                        print("No kf needed")
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

    # def detect(self, data, frame):
    #     image = self.image_list[self.tracked_frames]
    #     p_sorted, f_sorted, c_sorted = self.feature_loader.load_points_features_colors(data,image)
    #     if p_sorted is None or f_sorted is None or c_sorted is None:
    #         p_unmasked, f_unmasked, c_unmasked = features.extract_features(
    #             data.load_image(image), data.config)

    #         fmask = self.data.load_features_mask(image, p_unmasked)

    #         p_unsorted = p_unmasked[fmask]
    #         f_unsorted = f_unmasked[fmask]
    #         c_unsorted = c_unmasked[fmask]

    #         if len(p_unsorted) == 0:
    #             logger.warning('No features found in image {}'.format(image))
    #             return

    #         size = p_unsorted[:, 2]
    #         order = np.argsort(size)
    #         p_sorted = p_unsorted[order, :]
    #         f_sorted = f_unsorted[order, :]
    #         c_sorted = c_unsorted[order, :]
    #         data.save_features(image, p_sorted, f_sorted, c_sorted)

