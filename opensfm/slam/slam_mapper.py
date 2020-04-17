import numpy as np
import networkx as nx
from opensfm import pymap
from opensfm import pyslam
from opensfm import features
from opensfm import types
from opensfm import reconstruction
import slam_utils
import slam_debug
import logging
logger = logging.getLogger(__name__)

from collections import defaultdict


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
        self.num_tracked_lms = 0
        self.num_tracked_lms_thr = 15
        self.lms_ratio_thr = 0.9
        self.fresh_landmarks = []

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
            print("Updating velocity: T_{},W * T_W,{}".format(shot.id, self.last_shot.id))
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

    def new_keyframe_is_needed(self, shot: pymap.Shot):
        num_keyfrms = len(self.keyframes)
        min_obs_thr = 3 if (3 <= num_keyfrms) else 2
        last_kf: pymap.Shot = self.keyframes[-1]
        
        num_reliable_lms = last_kf.compute_num_valid_pts(min_obs_thr)
        max_num_frms_ = 10  # the fps
        min_num_frms_ = 2
        # if frame.frame_id > 15 and frame.frame_id % 3:
        #     return True
        frm_id_of_last_keyfrm_ = self.curr_kf.id
        print("curr_kf: ", self.curr_kf.id, num_keyfrms)
        print("frame.frame_id: ", shot.id, frm_id_of_last_keyfrm_)
        # frame.id
        # ## mapping: Whether is processing
        # #const bool mapper_is_idle = mapper_->get_keyframe_acceptability();
        # Condition A1: Add keyframes if max_num_frames_ or more have passed

        # since the last keyframe insertion
        cond_a1 = (frm_id_of_last_keyfrm_ + max_num_frms_ <= shot.id)
        # Condition A2: Add keyframe if min_num_frames_ or more has passed
        # and mapping module is in standby state
        cond_a2 = (frm_id_of_last_keyfrm_ + min_num_frms_ <= shot.id)
        # cond_a2 = False
        # Condition A3: Add a key frame if the viewpoint has moved from the
        # previous key frame
        cond_a3 = self.num_tracked_lms < (num_reliable_lms * 0.25)

        print("self.num_tracked_lms_thr {} self.num_tracked_lms {}\n \
               num_reliable_lms {} * self.lms_ratio_th={}".
              format(self.num_tracked_lms_thr, self.num_tracked_lms,
                     num_reliable_lms, num_reliable_lms * self.lms_ratio_thr))
        # Condition B: (Requirement for adding keyframes)
        # Add a keyframe if 3D points are observed above the threshold and
        # the percentage of 3D points is below a certain percentage
        cond_b = (self.num_tracked_lms_thr <= self.num_tracked_lms) and \
                 (self.num_tracked_lms < num_reliable_lms * self.lms_ratio_thr)

        print("cond_a1: {}, cond_a2: {}, cond_a3: {}, cond_b: {}"
              .format(cond_a1, cond_a2, cond_a3, cond_b))

        # Do not add if B is not satisfied
        if not cond_b:
            print("not cond_b -> no kf")
            return False
        
        # Do not add if none of A is satisfied
        if not cond_a1 and not cond_a2 and not cond_a3:
            print("not cond_a1 and not cond_a2 and not cond_a3 -> no kf")
            return False
        print("NEW KF", shot.name)
        return True

    def remove_redundant_kfs(self):
        pass
    def remove_redundant_lms(self):
        pass
    def insert_new_keyframe(self, shot: pymap.Shot):
        # Create new Keyframe
        self.add_keyframe(shot)
        # Now, we have to add the lms as observations
        lm_idc = shot.get_valid_landmarks_and_indices()
        scale_levels = self.extractor.get_scale_levels()
        for lm, idx in lm_idc:
            # If observed correctly, check it!
            # Triggers only for replaced landmarks
            if lm.is_observed_in_shot(shot):
                self.fresh_landmarks(lm)
            else:
                # add observation it
                self.map.add_observation(shot, lm, idx)
                # compute geometry
                pyslam.compute_descriptor(lm)
                pyslam.compute_normal_and_depth(lm, scale_levels)
        
        # Update connection
        shot.slam_data.update_graph_node()

        # self.slam_map_cleaner.update_lms_after_kf_insert(new_kf.ckf)
        self.remove_redundant_lms()


        # self.slam_map_cleaner.remove_redundant_lms(new_kf.kf_id)
        print("create_new_landmarks_before")
        chrono = reconstruction.Chronometer()
        self.create_new_landmarks()
        chrono.lap("create_landmarks")
        print("create_new_landmarks_after")
        self.update_new_keyframe()
        chrono.lap("update_keyframe")
        if self.n_keyframes % self.config_slam["run_local_ba_every_nth"] == 0:
            self.local_bundle_adjustment()
        chrono.lap("local_bundle_adjustment")
        slam_debug.avg_timings.addTimes(chrono.laps_dict)

        if self.n_keyframes % 50 == 0:
            chrono.start()
            self.create_reconstruction()
            self.save_reconstruction(shot.name + "aft")
            chrono.lap("create+save rec")
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
        chrono.start()
        # n_kf_removed = self.remove_redundant_kfs()
        # n_kf_removed = self.slam_map_cleaner.remove_redundant_kfs(new_kf.ckf, self.c_keyframes[0].kf_id)
        self.remove_redundant_kfs()
        n_kf_removed = 0
        print("Removed {} keyframes ".format(n_kf_removed))
        if (n_kf_removed > 0):
            print("Finally removed frames")
        chrono.lap("remove_redundant_kfs")
        slam_debug.avg_timings.addTimes(chrono.laps_dict)

    def create_new_landmarks(self):
        """Creates a new landmarks with using the newly added KF
        """
        # new_kf = self.c_keyframes[-1]
        new_kf = self.keyframes[-1]
        new_im = self.data.load_image(new_kf.name)
        kf_pose: pymap.Pose = new_kf.get_pose()
        new_cam_center = kf_pose.get_origin()
        new_Tcw = new_kf.get_world_to_cam()
        new_R = new_Tcw[0:3, 0:3]
        new_t = new_Tcw[0:3, 3]
        # Again, just take the last 10 frames
        # but not the current one!
        # num_covisibilities = 10
        # TODO: replace "local" keyframes by that
        # cov_kfs = new_kf.get_graph_node().get_top_n_covisibilities(2 * num_covisibilities)
        local_keyframes = self.keyframes[-5:-1]
        print("local_kf: ", len(local_keyframes))

        # TODO! check new_kf pose
        # for (old_kf, old_kf_py) in zip(local_keyframes, py_kfs):
        n_baseline_reject = 0
        chrono = slam_debug.Chronometer()
        # new_med_depth = new_kf.compute_median_depth(True)
        min_d, max_d = pyslam.SlamUtilities.compute_min_max_depth(new_kf)
        # min_d *= 0.5
        # max_d *= 2
        for old_kf in local_keyframes:
            old_cam_center = old_kf.get_origin()
            baseline_vec = old_cam_center - new_cam_center
            baseline_dist = np.linalg.norm(baseline_vec)
            median_depth_in_old = old_kf.compute_median_depth(True)
            if baseline_dist < 0.02 * median_depth_in_old:
                n_baseline_reject += 1
                continue

            # Compute essential matrix!
            old_Tcw = old_kf.get_world_to_camera()
            old_R = old_Tcw[0:3, 0:3]
            old_t = old_Tcw[0:3, 3]
            
            chrono.start()
            pyslam.SlamUtilities.cr
            E_old_to_new = self.guided_matcher.create_E_21(new_R, new_t, old_R, old_t)
            chrono.lap("compute E")
            matches = self.guided_matcher.match_for_triangulation_epipolar(new_kf, old_kf, E_old_to_new, min_d, max_d, False, 10)
            chrono.lap("match_for_triangulation_line_10")
            old_im = self.data.load_image(old_kf.im_name)
            self.triangulate_from_two_kfs(new_kf, old_kf, matches)
            chrono.lap("triangulate_from_two_kfs")
            slam_debug.avg_timings.addTimes(chrono.laps_dict)
        print("n_baseline_reject: ", n_baseline_reject)

    def triangulate_from_two_kfs(self, new_kf: pymap.Shot, old_kf: pymap.Shot, matches):
        # TODO: try without tracks graph
        frame1 = new_kf.name
        frame2 = old_kf.name
        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(frame1), bipartite=0)
        tracks_graph.add_node(str(frame2), bipartite=0)
        f_processed = defaultdict(int)
        pts1 = pyslam.SlamUtilities.keypts_from_shot(old_kf)
        p1, _, _ = features.\
            normalize_features(pts1, None, None,
                               self.camera[1].width, self.camera[1].height)
        
        pts2 = pyslam.SlamUtilities.keypts_from_shot(old_kf)
        p2, _, _ = features.\
            normalize_features(pts2, None, None,
                               self.camera[1].width, self.camera[1].height)

        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            # this checks whether the current kf was matched
            # to one of the landmarks.
            # if f2 is already in a lm
            f_processed[f1_id] += 1
            if f_processed[f1_id] > 1:
                print("double add!!")
                exit()
            x, y, s = p2[f2_id, 0:3]
            r, g, b = [0 , 0, 0] #c2[f2_id, :]
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(frame2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))

            x, y, s = p1[f1_id, 0:3]
            r, g, b = [0, 0, 0] #c1[f1_id, :]
            tracks_graph.add_edge(str(frame1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))
        # chrono.lap("track_graph")
        cameras = self.data.load_camera_models()
        camera = next(iter(cameras.values()))
        rec_tri = types.Reconstruction()
        rec_tri.reference = self.data.load_reference()
        rec_tri.cameras = cameras
        pose1 = slam_utils.mat_to_pose(new_kf.get_Tcw())
        shot1 = types.Shot()
        shot1.id = frame1
        shot1.camera = camera
        shot1.pose = pose1
        shot1.metadata = reconstruction.get_image_metadata(self.data, frame1)
        rec_tri.add_shot(shot1)

        pose2 = slam_utils.mat_to_pose(old_kf.get_Tcw())
        shot2 = types.Shot()
        shot2.id = frame2
        shot2.camera = camera
        shot2.pose = pose2
        shot2.metadata = reconstruction.get_image_metadata(self.data, frame2)
        rec_tri.add_shot(shot2)

        graph_inliers = nx.Graph()
        # chrono.lap("ba setup")
        np_before = len(rec_tri.points)
        reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
                                                 rec_tri, frame1,
                                                 self.data.config)
        np_after = len(rec_tri.points)
        print("Successfully triangulated {} out of {} points.".
              format(np_after, np_before))
        # chrono.lap("triangulateion")
        # edges1 = graph_inliers.edges(frame1)
        points = rec_tri.points
        points3D = np.zeros((len(points), 3))
        for idx, pt3D in enumerate(points.values()):
            points3D[idx, :] = pt3D.coordinates

        slam_debug.reproject_landmarks(points3D, None, slam_utils.mat_to_pose(new_kf.get_Tcw()), self.data.load_image(new_kf.im_name), self.camera[1], do_show=False)
        slam_debug.reproject_landmarks(points3D, None, slam_utils.mat_to_pose(old_kf.get_Tcw()), self.data.load_image(old_kf.im_name), self.camera[1], do_show=True)
        kf1 = new_kf
        kf2 = old_kf
        # Add to graph -> or better just create clm
        for _, gi_lm_id in graph_inliers.edges(frame1):
            # TODO: Write something like create_landmark
            pos_w = rec_tri.points[gi_lm_id].coordinates
            clm = self.slam_map.create_new_lm(kf2, pos_w)
            self.c_landmarks[clm.lm_id] = clm
            e1 = graph_inliers.get_edge_data(frame1, gi_lm_id)
            e2 = graph_inliers.get_edge_data(frame2, gi_lm_id)
            f1_id = e1['feature_id']
            f2_id = e2['feature_id']
            # connect landmark -> kf
            clm.add_observation(kf1, f1_id)
            clm.add_observation(kf2, f2_id)
            kf1.add_landmark(clm, f1_id)
            kf2.add_landmark(clm, f2_id)
            clm.compute_descriptor()
            clm.update_normal_and_depth()
            self.last_frame.cframe.add_landmark(clm, f1_id)
            self.slam_map_cleaner.add_landmark(clm)
