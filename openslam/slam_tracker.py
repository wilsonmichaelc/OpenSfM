import datetime
import logging
import numpy as np
import cv2
from opensfm import types
from opensfm import csfm
from opensfm.reconstruction import Chronometer
from opensfm import reconstruction
from opensfm import feature_loader
from opensfm import features

# from slam_matcher import SlamMatcher
import slam_matcher
from slam_mapper import SlamMapper
from slam_types import Frame
import slam_utils
import slam_debug
from itertools import compress
logger = logging.getLogger(__name__)

import cslam
class SlamTracker(object):

    # def __init__(self, data, config, guided_matcher):
    def __init__(self, guided_matcher):
        print("init slam tracker")
        self.guided_matcher = guided_matcher
        self.scale_factors = None
        self.num_tracked_lms = 0

    def bundle_tracking(self, points3D, observations, init_pose, camera,
                        config, data):
        """Estimates the 6 DOF pose with respect to 3D points

        Reprojects 3D points to the image plane and minimizes the
        reprojection error to the corresponding observations to 
        find the relative motion.
     
        Args:
            points3D: 3D points to reproject
            observations: their 2D correspondences
            init_pose: initial pose depending on the coord. system of points3D
            camera: intrinsic camera parameters
            config, data

        Returns:
            pose: The estimated (relative) 6 DOF pose
        """
        if len(points3D) != len(observations):
            print("len(points3D) != len(observations): ",
                  len(points3D), len(observations))
            return None
        # reproject_landmarks(points3D, observations, init_pose, camera, data)
        # match "last frame" to "current frame"
        # last frame could be reference frame
        # somehow match world points/landmarks seen in last frame
        # to feature matches
        fix_cameras = True
        chrono = Chronometer()
        ba = csfm.BundleAdjuster()
        # for camera in reconstruction.cameras.values():
        reconstruction.\
            _add_camera_to_bundle(ba, camera[1], camera[1], fix_cameras)
        
        # constant motion velocity -> just say id
        shot_id = str(0)
        camera_id = str(camera[0])
        camera_const = False
        ba.add_shot(shot_id, str(camera_id), init_pose.rotation,
                    init_pose.translation, camera_const)
        points_3D_constant = True
        # Add points in world coordinates
        for (pt_id, pt_coord) in enumerate(points3D):
            ba.add_point(str(pt_id), pt_coord, points_3D_constant)
            ft = observations[pt_id, :]
            ba.add_point_projection_observation(shot_id, str(pt_id),
                                                ft[0], ft[1], ft[2])
        # Assume observations N x 3 (x,y,s)
        ba.add_absolute_up_vector(shot_id, [0, 0, -1], 1e-3)
        ba.set_point_projection_loss_function(config['loss_function'],
                                              config['loss_function_threshold'])
        ba.set_internal_parameters_prior_sd(
            config['exif_focal_sd'],
            config['principal_point_sd'],
            config['radial_distorsion_k1_sd'],
            config['radial_distorsion_k2_sd'],
            config['radial_distorsion_p1_sd'],
            config['radial_distorsion_p2_sd'],
            config['radial_distorsion_k3_sd'])
        ba.set_num_threads(config['processes'])
        ba.set_max_num_iterations(50)
        ba.set_linear_solver_type("SPARSE_SCHUR")
        chrono.lap('setup')
        ba.run()
        chrono.lap('run_track')
        s = ba.get_shot(shot_id)
        pose = types.Pose()
        pose.rotation = [s.r[0], s.r[1], s.r[2]]
        pose.translation = [s.t[0], s.t[1], s.t[2]]
        valid_pts = self.discard_outliers(ba, len(points3D), pose, camera[1])
        # print("valid_pts!: ", valid_pts)
        return pose, valid_pts

    def discard_outliers(self, ba, n_pts, pose, camera):
        """Remove landmarks with large reprojection error
        or if reprojections are out of bounds
        """
        pts_outside = 0
        pts_inside = 0
        pts_outside_new = 0
        th = 0.006
        valid_pts = np.zeros(n_pts, dtype=bool)
        w, h = camera.width, camera.height
        for pt_id in range(0, n_pts):
            p = ba.get_point(str(pt_id))
            error = p.reprojection_errors['0']
            # Discard if reprojection error too large
            if np.linalg.norm(error) > th:
                pts_outside += 1
            else:
                # check if OOB
                camera_point = pose.transform([p.p[0], p.p[1], p.p[2]])
                if camera_point[2] <= 0.0:
                    pts_outside += 1
                    pts_outside_new += 1
                    continue
                point2D = camera.project(camera_point)
                if slam_utils.in_image(point2D, w, h):
                    pts_inside += 1
                    valid_pts[pt_id] = True
                else:
                    pts_outside += 1
                    pts_outside_new += 1

        print("pts inside {} and outside {}/ {}".
              format(pts_inside, pts_outside, pts_outside_new))
        return valid_pts

    def track_motion(self, slam_mapper: SlamMapper, frame: Frame,
                     init_pose, camera, config, data):
        """Estimate 6 DOF world pose of frame
        Reproject the landmarks seen in the last frame
        to frame and estimate the relative 6 DOF motion between
        the two by minimizing the reprojection error.
        """
        print("track_motion: ", slam_mapper.last_frame.im_name, "<->",
              frame.im_name)
        # TODO: Make an actual update on the closest frames in the map
        # For now, simply take the last 10 keyframes
        # return
        
        margin = 20
        # pose = np.vstack((slam_mapper.last_frame.world_pose.get_Rt(), np.array([0, 0, 0, 1])))
        # slam_mapper.init_frame.world_pose.compose()
        
        # Test the velocity
        # slam_mapper.update_with_last_frame(frame)
        # We need inverse last frame pose -> T_w_c
        # and T_c1_c0 = T_c1_w inv(T_c0_w)
        # pose_init =\
            # slam_mapper.last_frame.world_pose.compose(slam_mapper.last_frame.world_pose.compose(slam_mapper.init_frame.world_pose.inverse()))
        
        g = slam_mapper.graph
        r = slam_mapper.reconstruction
        init_frame = slam_mapper.pre_last
        last_frame = slam_mapper.last_frame
        velocity = last_frame.world_pose.compose(init_frame.world_pose.inverse()) # T_c1,w * inv(T_c0,w)
        slam_mapper.velocity
        DO_DEBUG = False
        if DO_DEBUG:
            print("frame: ", init_frame.im_name, last_frame.im_name)
            # debug
            p0_3D = np.zeros([len(g[slam_mapper.init_frame.im_name]), 3], dtype=np.float32)
            # p0 = np.zeros([len(g[last_frame.im_name]), 2], dtype=np.float32)
            for idx, lm_id in enumerate(g[slam_mapper.init_frame.im_name]):
                p0_3D[idx, :] = r.points[str(lm_id)].coordinates
                # p0[idx, :] = g.get_edge_data(last_frame.im_name, str(lm_id))['feature']
            im1, im2 = data.load_image(last_frame.im_name), data.load_image(frame.im_name)
            im3 = data.load_image(init_frame.im_name)
            slam_debug.disable_debug = False
            # camera.project_many(last_frame.world_pose.transform(p0_3D))
            slam_debug.reproject_landmarks(p0_3D, None, last_frame.world_pose, im1, camera[1],title="last frame", do_show=False)
            slam_debug.reproject_landmarks(p0_3D, None, init_frame.world_pose, im3, camera[1],title="init frame",do_show=False)
            velocity = last_frame.world_pose.compose(init_frame.world_pose.inverse())
            init = velocity.compose(last_frame.world_pose)
            slam_debug.reproject_landmarks(p0_3D, None, last_frame.world_pose, im2, camera[1],title="curr frame with last",do_show=False)
            slam_debug.reproject_landmarks(p0_3D, None, init, im2, camera[1],title="curr frame with vel",do_show=True)

            slam_debug.disable_debug = True
            # velocity = last_frame
            # slam_debug.reproject_landmarks(p0_3D, None, init_frame.world_pose, im3, camera,do_show=True)

        #     # project landmarks into kf1
        #     cam  = camera[1]
        #     slam_debug.disable_debug = False
        #     # # camera_point = self.init_frame.world_pose.transform_many(p0_3D)
        #     # camera_point = frame.world_pose.transform_many(p0_3D)
        #     # p1 = cam.project_many(camera_point)
        #     # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        #     # slam_debug.visualize_matches_pts(p0, p1, np.column_stack((a, a)), im1, im2, False, title="to frame"+frame.im_name)
        #     # # project landmarks into kf2
        #     # camera_point2 = last_frame.world_pose.transform_many(p0_3D)
        #     # p12 = cam.project_many(camera_point2)
        #     # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        #     # slam_debug.visualize_matches_pts(p0, p12, np.column_stack((a, a)), im1, im3, False, title="to last frame"+last_frame.im_name)
        #     # # project landmarks into coordinate system of kf 1 and then to kf2
        #     camera_point3 = velocity.compose(last_frame.world_pose).transform_many(p0_3D)
        #     p13 = cam.project_many(camera_point3)
        #     a = np.asarray(np.arange(0,len(p0)), dtype=int)
        #     slam_debug.visualize_matches_pts(p0, p13, np.column_stack((a, a)), im1, im2, True, title="constant velocity")
        #     # debug end
        #     slam_debug.disable_debug = True
        
        pose_init = velocity.compose(last_frame.world_pose) # slam_mapper.last_frame.world_pose.compose(slam_mapper.velocity)
        pose_init = np.vstack((pose_init.get_Rt(), np.array([0, 0, 0, 1])))
        frame.cframe.set_pose(np.linalg.inv(pose_init))
        # cslam.SlamDebug.reproject_last_lms(last_frame.cframe, frame.cframe, im1, im2)

        matches = self.guided_matcher.\
            match_current_and_last_frame(frame.cframe, slam_mapper.last_frame.cframe, margin)
        im1, im2 = data.load_image(last_frame.im_name), data.load_image(frame.im_name)
        # cslam.SlamDebug.print_matches_from_lms(last_frame.cframe, frame.cframe, im1, im2)

        # if DO_DEBUG:
        #     im1, im2 = data.load_image(last_frame.im_name), data.load_image(frame.im_name)
        #     # slam_debug.visualize_matches_pts(last_frame.cframe.getKptsPy(), frame.cframe.getKptsPy(), np.column_stack((a, a)), im1, im2, False, title="to last frame and then frame")

        #     cslam.SlamDebug.print_matches_from_lms(last_frame.cframe, frame.cframe, 
        #                                         im1, im2)
            #plot matches
            # slam_debug.disable_debug = False
            # slam_debug.visualize_matches_pts(p0, p13, np.column_stack((a, a)), im1, im2, True, title="to last frame and then frame")
            # slam_debug.disable_debug = True
        
        # frame.cframe.parent_kf = slam_mapper.c_keyframes[-1]
        n_matches = len(matches)
        if n_matches < 10: # not enough matches found, increase margin
            matches = self.guided_matcher.\
                match_current_and_last_frame(frame.cframe, slam_mapper.last_frame.cframe, margin*2)
            if n_matches < 10:
                logger.error("Tracking lost!!")
                exit()

        # (lms, points2D) = frame.cframe.get_lm_and_obs()
        lms = frame.cframe.get_valid_lms()
        points2D = frame.cframe.get_valid_keypts()
        valid_ids = frame.cframe.get_valid_idx()
        print("got: ", len(lms), " landmarks and ", len(points2D))
        
        # normalize
        points2D, _, _ = features.\
            normalize_features(points2D, None, None, camera[1].width, camera[1].height)
        # print("points2D: ", points2D)

        points3D = np.zeros((len(lms), 3), dtype=np.float)
        for i, lm in enumerate(lms):
            points3D[i, :] = lm.get_pos_in_world()

        # Set up bundle adjustment problem
        pose, valid_pts = self.bundle_tracking(points3D, points2D, init_pose,
                                               camera, config, data)

        # frame.cframe.set_outlier(np.array(valid_ids)[not valid_pts])
        frame.cframe.set_outlier(np.array(valid_ids)[np.invert(valid_pts)])
        # invalid_ids = g
        # frame.cframe.set_outlier()
        # discard outlier matches
        num_valid_matches = frame.cframe.discard_outliers()
        if num_valid_matches < 10:
            logger.error("Tracking lost!!")
            exit()
        return pose
        # point3D, points2D = cslam.GuidedMatcher.get_point_lm_correspondences()

        # local_landmarks = self.guided_matcher.\
        #     update_local_landmarks(slam_mapper.c_keyframes[-10:],
        #                            frame.frame_id)
        # n_matches = self.guided_matcher.\
        #     match_frame_and_landmarks(self.scale_factors,
        #                               frame.cframe, local_landmarks, 10.0)

        # chrono = Chronometer()
        # margin = 10
        # matches = slam_matcher.\
        #     match_frame_to_landmarks(frame.descriptors,
        #                              slam_mapper.last_frame.landmarks_,
        #                              margin, data, slam_mapper.graph)
        # chrono.lap('matching')
        # matches_test = slam_matcher.match_desc_desc(frame.descriptors, 
        #                                             slam_mapper.curr_kf.descriptors, data)
        # print("matches_test: ", matches_test)
        # # if len(matches) < 30:
        # #     print("Not enough matches!", len(matches))
        # #     return None
        # matches = np.asarray(matches)
        # print("track_motion matches: ", matches.shape)
        # landmarks = [slam_mapper.last_frame.landmarks_[m1] for m1 in matches[:, 1]]
        # print("n_matches: ", len(matches))
        # points3D = np.zeros((len(landmarks), 3))
        # points = slam_mapper.reconstruction.points
        # for idx, lm_id in enumerate(landmarks):
        #     points3D[idx, :] = points[lm_id].coordinates
        #     frame.landmarks_.append(lm_id)
        # print("len(last_frame.landmarks_): ",
        #       len(slam_mapper.last_frame.landmarks_),
        #       "len(landmarks): ", len(frame.landmarks_),
        #       "len(points3D): ", len(points3D))
        # points2D, _, _ = frame.load_points_desc_colors()
        # print("points2D.shape: ", points2D.shape)
        # points2D = points2D[matches[:, 0], :]
        # chrono.lap("dummy")
        # # Set up bundle adjustment problem
        # pose, valid_pts = self.bundle_tracking(points3D, points2D, init_pose,
        #                                        camera, config, data)
        # chrono.lap("tracking")
        # print("tracker f2: ", frame.im_name, len(frame.landmarks_))
        # frame.landmarks_ = list(compress(frame.landmarks_, valid_pts))
        # print("tracker f3: ", frame.im_name, len(frame.landmarks_))
        # print("Tracking times: ", chrono.lap_times())
        # slam_debug.\
        #     visualize_tracked_lms(points2D[np.array(valid_pts, dtype=bool), :], frame, data)
        # return pose

    def track(self, slam_mapper: SlamMapper, frame: Frame, config, camera,
              data):
        """Tracks the current frame with respect to the reconstruction
        """

        """ last_frame, frame, camera, init_pose, config, data):
        Align the current frame to the already estimated landmarks
            (visible in the last frame)
            landmarks visible in last frame
        """

        # Try to match to last frame first
        init_pose = slam_mapper.last_frame.world_pose
        pose_tracking = self.track_motion(slam_mapper, frame,
                                 init_pose, camera, config, data)

        local_landmarks = self.guided_matcher.update_local_landmarks(slam_mapper.c_keyframes[-10:], frame.frame_id)
        frame.cframe.set_pose(np.linalg.inv(np.vstack((pose_tracking.get_Rt(), [0, 0, 0, 1]))))
        # print(local_landmarks)
        n_matches = self.guided_matcher.search_local_landmarks(local_landmarks, frame.cframe)
        # Now, local optimization
        print("n_matches {} found in current frame.".format(n_matches))
        idx = 0
        valid_lms = frame.cframe.get_valid_lms()
        valid_kps = frame.cframe.get_valid_keypts()
        valid_ids = frame.cframe.get_valid_idx()
        points3D = np.zeros((len(valid_lms), 3))
        print("n_matches: ", n_matches, " len: ", len(valid_lms))
        for idx, lm in enumerate(valid_lms):
            points3D[idx, :] = lm.get_pos_in_world()
        # observations = valid_kps[]

        # observations = valid_kps
        observations, _, _ = features.normalize_features(valid_kps, None, None, camera[1].width, camera[1].height)

        #TODO: Remove debug stuff
        # slam_debug.disable_debug = False
        # slam_debug.reproject_landmarks(points3D, observations, pose_tracking, 
        #                                frame.image, camera[1],
        #                                title="bef tracking: "+frame.im_name, obs_normalized=True, do_show=False)
        pose, valid_pts = self.\
            bundle_tracking(points3D, observations, frame.world_pose, camera, data.config, data)

        # print("pose after! ", pose.rotation, pose.translation)
        # print("valid_pts: ", len(valid_pts), " vs ", len(observations))
        # slam_debug.reproject_landmarks(points3D, observations,
        #                                pose, frame.image, camera[1],
        #                                title="aft tracking: "+frame.im_name, obs_normalized=True, do_show=True)
        # slam_debug.disable_debug = True

        # for 

                
        # print("got: ", len(lms), " landmarks and ", len(points2D))
        
        self.num_tracked_lms = np.sum(valid_pts)
        # frame.cframe.set_outlier(np.array(valid_ids)[not valid_pts])
        frame.cframe.set_outlier(np.array(valid_ids)[np.invert(valid_pts)])
        n_tracked = frame.cframe.clean_and_tick_landmarks()
        print("n tracked: ", n_tracked)
        return pose
