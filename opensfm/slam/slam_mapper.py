import numpy as np
from opensfm import pymap
from opensfm import pyslam 

import logging
logger = logging.getLogger(__name__)

class SlamMapper(object):
    def __init__(self, data, config_slam, camera, slam_map, extractor):
        self.data = data
        self.camera = camera
        self.config = data.config
        self.config_slam = config_slam
        self.map = slam_map
        self.keyframes = []
        self.n_keyframes = 0
        self.n_frames = 0
        self.curr_kf = None
        self.last_shot = None
        self.pre_last = None
        self.extractor = extractor


    def add_keyframe(self, kf):
        """Adds a keyframe to the map graph
        and the covisibility graph
        """
        logger.debug("Adding new keyframe # {}, {}".format(kf.id, kf.name))
        self.n_keyframes += 1
        self.keyframes.append(kf)
        self.curr_kf = kf

    def update_with_last_frame(self, shot: pymap.Shot):
        """Updates the last frame and the related variables in slam mapper
        """
        if self.n_frames > 0:  # we alread have frames
            self.velocity = shot.get_pose().get_world_to_cam().dot(
                self.last_shot.get_pose().get_cam_to_world())
            # self.velocity = frame.world_pose.compose(self.last_frame.world_pose.inverse())
            self.pre_last = self.last_shot
        self.n_frames += 1
        self.last_shot = shot

    def create_init_map(self, graph_inliers, rec_init,
                        init_shot: pymap.Shot, curr_shot: pymap.Shot,
                        init_pdc=None, other_pdc=None):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
        """
        kf1 = init_shot
        kf2 = curr_shot
        pose1, pose2 = pymap.Pose(), pymap.Pose()
        pose1.set_from_world_to_cam(
            np.vstack((rec_init.shots[kf1.name].pose.get_Rt(),
                       np.array([0, 0, 0, 1]))))
        pose2.set_from_world_to_cam(
            np.vstack((rec_init.shots[kf2.name].pose.get_Rt(),
                       np.array([0, 0, 0, 1]))))
        kf1.set_pose(pose1)
        kf2.set_pose(pose2)
        # Add to data and covisibility
        self.add_keyframe(kf1)
        self.add_keyframe(kf2)
        self.update_with_last_frame(kf1)
        self.update_with_last_frame(kf2)
        for lm_id in graph_inliers[kf1.name]:
            pos_w = rec_init.points[str(lm_id)].coordinates
            lm = self.map.create_landmark(int(lm_id), pos_w)
            lm.set_ref_shot(kf1)
            f1_id = graph_inliers.\
                get_edge_data(lm_id, kf1.name)["feature_id"]
            f2_id = graph_inliers.\
                get_edge_data(lm_id, kf2.name)["feature_id"]
            # connect landmark -> kf
            self.map.add_observation(kf1, lm, f1_id)
            self.map.add_observation(kf2, lm, f2_id)
            pyslam.SlamUtilities.compute_descriptor(lm)
            pyslam.SlamUtilities.compute_normal_and_depth(
                lm, self.extractor.get_scale_levels())
        print("create_init_map: len(local_landmarks): ",
              self.map.number_of_landmarks())
        # Change that according to cam model
        median_depth = kf1.compute_median_depth(False)
        min_num_triangulated = 100
        # print("curr_kf.world_pose: ", curr_kf.world_pose.get_Rt)
        print("Tcw bef scale: ", kf2.get_pose().get_world_to_cam())
        if kf2.compute_num_valid_pts(1) < min_num_triangulated and median_depth < 0:
            logger.info("Something wrong in the initialization")
        else:
            scale = 1.0 / median_depth
            kf2.scale_pose(scale)
            kf2.scale_landmarks(scale)
            # self.slam_map.scale_map(kf1, kf2, 1.0 / median_depth)
        # curr_frame.world_pose = slam_utils.mat_to_pose(kf2.get_Tcw())
        print("Tcw aft scale: ", kf2.get_pose().get_world_to_cam())
        # curr_frame.world_pose = curr_kf.world_pose
        print("Finally finished scale")

