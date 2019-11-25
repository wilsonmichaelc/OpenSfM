import datetime
import logging

from opensfm import csfm
from opensfm.reconstruction import Chronometer
from opensfm import reconstruction

from slam_matcher import SlamMatcher


class SlamTracker(object):

    def __init__(self,  data, config):
        self.slam_matcher = SlamMatcher(config)
        print("init slam tracker")

    def track_reprojection(self, points3D, observations, init_pose, camera,
                           config, data):
        """Estimates the 6 DOF pose with respect to 3D points

        Reprojects 3D points to the image plane and minimizes the
        reprojection error to the correspondences.

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
            return False
        # match "last frame" to "current frame"
        # last frame could be reference frame
        # somehow match world points/landmarks seen in last frame
        # to feature matches
        fix_cameras = not config['optimize_camera_parameters']

        chrono = Chronometer()
        ba = csfm.BundleAdjuster()
        # for camera in reconstruction.cameras.values():
        reconstruction._add_camera_to_bundle(ba, camera, False)
        # init with a constant motion model!
        # shot == last image
        # shot = reconstruction.shots[last_frame]
        # r = shot.pose.rotation
        # t = shot.pose.translation
        # fix the world pose of the last_frame
        # ba.add_shot(shot.id, 0, r, t, True)

        # constant motion velocity -> just say id
        shot_id = 0
        camera_id = 0
        ba.add_shot(shot_id, camera_id, init_pose)
        points_3D_constant = True
        # Add points in world coordinates
        for (pt_id, pt_coord) in enumerate(points3D):
            print("point id: ", pt_id, " coord: ", pt_coord)
            ba.add_point(pt_id, pt_coord, points_3D_constant)
            ft = observations[pt_id, :]
            print("Adding obs: ", pt_id, ft)
            ba.add_point_projection_observation(shot_id, pt_id,
                                                ft[0], ft[1], ft[2])
        #Assume observations N x 3 (x,y,s)
        # for (ft_id, ft) in enumerate(observations):
        #     print("Adding: ", ft_id, ft)
        #     ba.add_point_projection_observation(shot_id, ft_id,
        #                                         ft[0], ft[1], ft[2])

        align_method = config['align_method']
        if align_method == 'auto':
            align_method = align.detect_alignment_constraints(config, reconstruction, gcp)
        if align_method == 'orientation_prior':
            if config['align_orientation_prior'] == 'vertical':
                for shot_id in reconstruction.shots:
                    ba.add_absolute_up_vector(shot_id, [0, 0, -1], 1e-3)
            if config['align_orientation_prior'] == 'horizontal':
                for shot_id in reconstruction.shots:
                    ba.add_absolute_up_vector(shot_id, [0, 1, 0], 1e-3)

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
        # for track in graph[shot_id]:
        #     #track = id of the 3D point
        #     if track in reconstruction.points:
        #         point = graph[shot_id][track]['feature']
        #         scale = graph[shot_id][track]['feature_scale']
        #         print("track: ", track, " shot_id: ", shot_id)
        #         print("point: ", point, " scale: ", scale)
        #         ba.add_point_projection_observation(
        #             shot_id, track, point[0], point[1], scale)
        
        # for shot_id in reconstruction.shots:
        #     if shot_id in graph:
        #         for track in graph[shot_id]:
        #             #track = id of the 3D point
        #             if track in reconstruction.points:
        #                 point = graph[shot_id][track]['feature']
        #                 scale = graph[shot_id][track]['feature_scale']
        #                 print("track: ", track, " shot_id: ", shot_id)
        #                 print("point: ", point, " scale: ", scale)
        #                 ba.add_point_projection_observation(
        #                     shot_id, track, point[0], point[1], scale)
        #now match

        
        chrono.lap('setup')
        ba.run()
        chrono.lap('run')

        #
        return True

    def track(self, graph, reconstruction, landmarks, last_frame, 
              frame, camera, init_pose, config, data):
        """Align the current frame to the already estimated landmarks
            (visible in the last frame)
            landmarks visible in last frame
        """
        m1, idx1, idx2 = self.slam_matcher.match_landmarks_to_image(
                            landmarks, frame, last_frame, camera, data)

        #prepare the bundle
        

        # tracks are the matched landmarks
        # match landmarks to current frame
        # last frame is typically != last keyframe
        # landmarks contain feature id in last frame
        
        #load feature so both frames
        # p1, f1, _ = 
        #landmarks = LandmarkStorage()

        # for landmark in landmarks:
            # feature_id = landmark.fid
            
        

        if n_matches < 100: # kind of random number
            return False

        # velocity = T_(N-1)_(N-2) pre last to last
        # init_pose = T_(N_1)_w * T_(N-1)_W * inv(T_(N_2)_W)
        # match "last frame" to "current frame"
        # last frame could be reference frame
        # somehow match world points/landmarks seen in last frame
        # to feature matches
        fix_cameras = not config['optimize_camera_parameters']

        #call bundle
        self.track_reprojection()
