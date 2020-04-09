import numpy as np
from opensfm import pymap
from opensfm import pyslam 

import logging
logger = logging.getLogger(__name__)

class SlamMapper(object):
    def __init__(self, data, config_slam, camera, slam_map):
        self.data = data
        self.camera = camera
        self.config = data.config
        self.config_slam = config_slam
        self.map = slam_map
        self.keyframes = []
        self.curr_kf = None

    def add_keyframe(self, kf):
        """Adds a keyframe to the map graph
        and the covisibility graph
        """
        logger.debug("Adding new keyframe # {}, {}".format(kf.id, kf.name))
        self.n_keyframes += 1
        self.keyframes.append(kf)
        self.curr_kf = kf

    def update_with_last_frame(self, shot: Shot):
        """Updates the last frame and the related variables in slam mapper
        """
        if self.n_frames > 0:  # we alread have frames
            self.velocity = frame.world_pose.compose(self.last_frame.world_pose.inverse())
            self.pre_last = self.last_frame
        self.n_frames += 1
        self.last_frame = frame

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
        pose1.set_from_world_to_cam(np.vstack((rec_init.shots[kf1.name].pose.get_Rt(), np.array([0, 0, 0, 1]))))
        pose2.set_from_world_to_cam(np.vstack((rec_init.shots[kf2.name].pose.get_Rt(), np.array([0, 0, 0, 1]))))
        # rec_init
        # init_frame.frame_id = 0
        # Create keyframes
        # init_frame.world_pose = rec_init.shots[init_frame.im_name].pose
        # self.init_frame = Keyframe(init_frame, self.data, 0)
        # curr_frame.frame_id = 1
        # curr_frame.world_pose = rec_init.shots[curr_frame.im_name].pose
        # curr_kf = Keyframe(curr_frame, self.data, 1)

        # self.init_frame.ckf = kf1
        # curr_kf.ckf = kf2

        # Add to data and covisibility
        self.add_keyframe(kf1)
        self.add_keyframe(kf2)
        self.update_with_last_frame(kf1)
        self.update_with_last_frame(kf2)
        for lm_id in graph_inliers[self.init_frame.im_name]:
            pos_w = rec_init.points[str(lm_id)].coordinates
            clm = self.slam_map.create_new_lm(kf1, pos_w)
            self.c_landmarks[clm.lm_id] = clm
            f1_id = graph_inliers.\
                get_edge_data(lm_id, self.init_frame.im_name)["feature_id"]
            f2_id = graph_inliers.\
                get_edge_data(lm_id, curr_kf.im_name)["feature_id"]
            # connect landmark -> kf
            clm.add_observation(kf1, f1_id)
            clm.add_observation(kf2, f2_id)
            # connect kf -> landmark in the graph
            kf1.add_landmark(clm, f1_id)
            kf2.add_landmark(clm, f2_id)
            clm.compute_descriptor()
            clm.update_normal_and_depth()
            curr_frame.cframe.add_landmark(clm, f2_id)
        print("create_init_map: len(local_landmarks): ",
              self.slam_map.get_num_landmarks())
        # Change that according to cam model
        median_depth = kf1.compute_median_depth(False)
        min_num_triangulated = 100
        # print("curr_kf.world_pose: ", curr_kf.world_pose.get_Rt)
        print("Tcw bef scale: ", kf2.get_Tcw())
        if kf2.get_num_tracked_lms(1) < min_num_triangulated and median_depth < 0:
            logger.info("Something wrong in the initialization")
        else:
            self.slam_map.scale_map(kf1, kf2, 1.0 / median_depth)
        curr_frame.world_pose = slam_utils.mat_to_pose(kf2.get_Tcw())
        print("Tcw aft scale: ", kf2.get_Tcw())
        # curr_frame.world_pose = curr_kf.world_pose
        print("Finally finished scale")

