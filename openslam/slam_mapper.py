from opensfm import types
from opensfm import reconstruction
from slam_types import Frame
from slam_types import Keyframe
from slam_types import Landmark
import slam_debug
import slam_utils
# from slam_tracker import SlamTracker
from slam_matcher import SlamMatcher
from collections import defaultdict
import networkx as nx
import logging
import numpy as np
logger = logging.getLogger(__name__)
from itertools import compress

class SlamMapper(object):

    def __init__(self, data, config, camera):
        """SlamMapper holds a local and global map
        """
        self.data = data
        self.camera = camera
        self.reconstruction = None
        self.last_frame = Frame("dummy", -1)
        # Threshold of the ratio of the number of 3D points observed in the
        # current frame to the number of 3D points observed in the latest KF
        self.num_tracked_lms_thr = 15
        self.lms_ratio_thr = 0.9
        self.n_tracks = 0
        self.graph = nx.Graph()
        self.reconstruction = []
        self.n_landmarks = 0  # == unique lm id
        self.n_keyframes = 0  # == unique kf id
        self.n_frames = 0     # == unique frame id
        self.curr_kf = None
        # dict because frames can be deleted
        self.keyframes = {}     # holds the id, Frame()
        self.frames = {}        # holds the id, Frame()
        # local map keyframes are the ones that have shared
        # landmarks with the current frame
        self.local_keyframes = []
        self.local_landmarks = []
        self.slam_matcher = SlamMatcher(config)
        self.covisibility = nx.Graph()
        self.fresh_landmarks = []
        self.current_lm_i = 0

    def estimate_pose(self):
        if self.curr_kf is not None:
            return self.curr_kf.world_pose
        return types.Pose()

    def create_init_map(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame):
        """Basically the graph contains
        the keyframes/shots and landmarks.
        Edges are connections between keyframes
        and landmarks.
        """
        # Store the initial graph and reconstruction
        self.graph = graph_inliers
        self.reconstruction = rec_init

        init_frame.frame_id = 0
        # Create keyframes
        self.init_frame = Keyframe(init_frame, self.data, 0)
        self.init_frame.world_pose = \
            rec_init.shots[init_frame.im_name].pose
        curr_frame.frame_id = 1
        # self.curr_kf = Keyframe(curr_frame, self.data, 1)
        # self.curr_kf.world_pose = rec_init.shots[curr_frame.im_name].pose
        curr_kf = Keyframe(curr_frame, self.data, 1)
        curr_kf.world_pose = rec_init.shots[curr_frame.im_name].pose

        # Add to data and covisibility
        self.add_keyframe(self.init_frame)
        self.add_keyframe(curr_kf)
        print("init_frame: ", self.init_frame.world_pose.get_origin(),
              self.init_frame.world_pose.get_rotation_matrix())
        print("curr_kf: ", curr_kf.world_pose.get_origin(),
              curr_kf.world_pose.get_rotation_matrix())

        self.n_frames = 2

        max_lm = 0  # find the highest lm id
        # Add landmark objects to nodes
        for lm_id in self.graph[self.init_frame.im_name]:
            lm = Landmark(int(lm_id))
            self.graph.add_node(lm_id, data=lm)

            int_id = int(lm_id)
            if int_id > max_lm:
                max_lm = int_id
            lm.compute_descriptor(self.graph)
            pos_w = rec_init.points[str(lm_id)].coordinates
            lm.update_normal_and_depth(pos_w, self.graph)
            self.local_landmarks.append(lm_id)
        self.current_lm_id = max_lm

        # also copy them to current kf
        curr_kf.landmarks_ = self.local_landmarks.copy()

        # copy local landmarks to last_frame
        self.last_frame.landmarks_ = curr_kf.landmarks_.copy()
        self.last_frame.im_name = curr_kf.im_name

        print("create_init_map with landmarks: ", len(curr_kf.landmarks_),
              len(self.last_frame.landmarks_), len(self.local_landmarks))
        # if self.system_initialized:
        # self.tracked_frames += 1
        self.update_local_map(curr_frame)
        self.mapping_with_new_keyframe(self.init_frame)
        self.mapping_with_new_keyframe(curr_kf)

    def add_keyframe(self, kf: Keyframe):
        """Adds a keyframe to the map graph
        and the covisibility graph
        """
        # add kf object to existing graph node
        self.graph.add_node(str(kf.im_name), bipartite=0, data=kf)
        self.covisibility.add_node(str(kf.im_name))
        self.n_keyframes += 1
        shot1 = types.Shot()
        shot1.id = kf.im_name
        shot1.camera = self.camera[1]
        print("kf.im_name: ", kf.im_name, "camera: ", self.camera)
        shot1.pose = kf.world_pose
        shot1.metadata = reconstruction.\
            get_image_metadata(self.data, kf.im_name)
        self.reconstruction.add_shot(shot1)

    def add_landmark(self, lm: Landmark):
        """Add landmark to graph"""
        # print("Add_landmark: ", str(lm.lm_id))
        self.graph.add_node(str(lm.lm_id), bipartite=1, data=lm)
    # def add_lm_kf(self, lm: Landmark, kf: Keyframe)
    #     self.graph.add_edge(str(lm.lm_id), str(lm_id)
    def erase_keyframe(self,  kf: Keyframe):
        self.graph.remove_node(kf.im_name)
        self.covisibility.remove_node(kf.im_name)

    def erase_landmark(self, lm: Landmark):
        self.graph.remove_node(lm.lm_id)

    def set_curr_kf(self, keyframe):
        """Sets a new keyframe

        Arguments:
            keyframe: of type Frame
        """
        self.curr_kf = keyframe
        self.n_keyframes += 1
        self.set_last_frame(keyframe)
        # TODO: Think about initializing the new keyframe with the
        #       old landmarks
        # if len(self.keyframes) > 0:
            # self.keyframes[-1].store()
        # self.keyframes.append(keyframe)
        self.keyframes[keyframe.id] = keyframe

    def set_last_frame(self, frame: Frame):
        """Sets the last frame

        Arguments:
            frame: of Frame
        """
        self.n_frames += 1
        print("set_last_frame 1: ", len(frame.landmarks_))
        self.last_frame = frame
        print("set_last_frame: ", frame.im_name, self.last_frame.im_name,
              len(frame.landmarks_), len(self.last_frame.landmarks_))
        print("set_last_frame: ", frame.frame_id, "/", self.n_frames)
        self.frames[frame.frame_id] = frame
        # self.frames.landmarks_ = frame.landmarks_

    def add_frame_to_reconstruction(self, frame, pose, camera, data):
        shot1 = types.Shot()
        shot1.id = frame
        print("add_frame_to_reconstructioncamera: ", camera)
        print("add_frame_to_reconstructioncamera: ", camera[1].id)
        print("add_frame_to_reconstruction frame: ", frame)
        shot1.camera = camera[1]
        shot1.pose = types.Pose(pose.rotation, pose.translation)
        shot1.metadata = reconstruction.get_image_metadata(data, frame)
        self.reconstruction.add_shot(shot1)

    def paint_reconstruction(self, data):
        reconstruction.paint_reconstruction(data, self.graph,
                                            self.reconstruction)

    def save_reconstruction(self, data, name: str):
        data.save_reconstruction([self.reconstruction],
                                 'reconstruction'+name+'.json')

    def clean_landmarks(self):
        return True
    
    def update_local_keyframes(self, frame: Frame):
        """Count number of lm shared between current frame and neighbour KFs
        (count obs.). For each keyframe, we keep count of how many lms it
        shares with the current one.
        """
        print("update_local_keyframes")
        kfs_weights = defaultdict(int)
        for lm_id in frame.landmarks_:
            # find the number of sharing landmarks between 
            # the current frame and each of the neighbor keyframes
            connected_kfs = self.graph[lm_id]
            for kfs in connected_kfs:
                kfs_weights[kfs] += 1
        
        print("kfs_weights: ", kfs_weights, len(kfs_weights))
        if len(kfs_weights) == 0:
            return
        
        # kfs_weights = sorted(kfs_weights)
        self.local_keyframes.clear()
        max_weight = 0
        nearest_frame = -1
        for kf_id, weight in kfs_weights.items():
            self.local_keyframes.append(kf_id)
            kf: Keyframe = self.graph.node[kf_id]['data']
            kf.local_map_update_identifier = frame.frame_id
            if weight > max_weight:
                max_weight = weight
                self.nearest_covisibility = kf

        # max_local_keyframes = 60
        # add the second-order keyframes to the local landmarks
        # for local_kf in self.local_keyframes:
        #     if len(self.local_keyframes) > max_local_keyframes:
        #         break

    def update_local_landmarks(self, frame: Frame):
        """Update local landmarks by adding
        all the landmarks of the local keyframes.
        """
        # for lm in self.local_landmarks:
            # print("lm bef clear: ", lm)
        self.local_landmarks.clear()
        print("update_local_landmarks")
        for kf_id in self.local_keyframes:
            print("kf_id: ", kf_id)
            for lm_id in self.graph[kf_id]:
                # print("upd lm_node: ", lm_id, self.graph.node[str(lm_id)])
                if len(self.graph.nodes[str(lm_id)]) == 0:
                    print("Problem: ", lm_id)
                else:
                    lm = self.graph.node[str(lm_id)]['data']
                    # Avoid duplication
                    if lm.local_map_update_identifier == frame.frame_id:
                        continue
                    lm.local_map_update_identifier = frame.frame_id
                    self.local_landmarks.append(lm_id)

        # count the number of lmid
        lm_count = defaultdict(int)
        for lm in self.local_landmarks:
            lm_count[lm] += 1
        if len(lm_count) > 0:
            print("lm_count", max(lm_count.values()), len(lm_count))

    def apply_landmark_replace(self):
        print('apply landmark?')

    def set_local_landmarks(self):
        print("set_local_landmarks()")

    def update_local_map(self, frame: Frame):
        """Called after init and normal tracking
        """
        print("update_local_map for current frame: ",
              frame.frame_id, frame.im_name)
        # Todo: unify update_local_kf, keyframes and set
        self.update_local_keyframes(frame)
        self.update_local_landmarks(frame)
        # self.set_local_landmarks()
        # landmarks are already set
        return True

    def search_local_landmarks(self, frame: Frame):
        """ Acquire more 2D-3D matches by reprojecting the 
        local landmarks to the current frame
        """
        print("search_local_landmarks: ", len(frame.landmarks_))
        for lm_id in frame.landmarks_:
            lm = self.graph.node[lm_id]['data']
            lm.is_observable_in_tracking = False
            lm.identifier_in_local_lm_search_ = \
                frame.frame_id
            lm.num_observable += 1
        
        # found_candidate = False

        # for lm in self.local_keyframes:
            # if lm.identifier_in_local_lm_search_ == frame.frame_id:
                # continue
        # observations = self.observable_in_frame(frame)

        # print("Found {} observations".format(len(observations)))
        
        # acquire more 2D-3D matches by projecting the local landmarks to the current frame
        # match::projection projection_matcher(0.8);
        # const float margin = (curr_frm_.id_ < last_reloc_frm_id_ + 2)
        #                     ? 20.0 : ((camera_->setup_type_ == camera::setup_type_t::RGBD)
        #                             ? 10.0 : 5.0);
        # projection_matcher.match_frame_and_landmarks(curr_frm_, local_landmarks_, margin);
        margin = 5
        print("self.local_landmarks: ", len(self.local_landmarks))
        matches = self.slam_matcher.\
            match_frame_to_landmarks(frame, self.local_landmarks, margin,
                                     self.data, self.graph)
        print("matches: ", len(matches))
        return matches

    def observable_in_frame(self, frame: Frame):
        """ Similar to frame.can_observe in OpenVSlam
        """
        pose_world_to_cam = frame.world_pose
        cam_center = frame.world_pose.get_origin()
        factor = self.camera[1].height/self.camera[1].width
        # found_candidate = False
        observations = []
        for lm_id in self.local_landmarks:
            lm = self.graph.node[lm_id]['data']
            if lm.identifier_in_local_lm_search_ == frame.frame_id:
                continue
            # check if observeable
            p = self.reconstruction.points[str(lm.lm_id)].coordinates
            # print("p: ", p)
            camera_point = pose_world_to_cam.transform(p)
            print("camera_point", camera_point)
            if camera_point[2] <= 0.0:
                continue
            point2D = self.camera[1].project(camera_point)
            # print("point2D: ", point2D)
            is_in_image = slam_utils.in_image(point2D, factor)
            # print("point2D: ", point2D, factor, is_in_image)
            #TODO: check boundaries?
            cam_to_lm_vec = p - cam_center
            cam_to_lm_dist = np.linalg.norm(cam_to_lm_vec)

            #TODO: Check feature scale?
            # Compute normal
            lm.update_normal_and_depth(p, self.graph)
            mean_normal = lm.mean_normal
            ray_cos = np.dot(cam_to_lm_vec, mean_normal)/cam_to_lm_dist
            if ray_cos < 0.5:
                continue
            observations.append(point2D)
            # found_candidate = True

            #TODO: scale_level
            # pred_scale_lvl = lm.predict_scale_level(dist, )

            # return True, point2D
        return observations

    # OpenVSlam mapping module
    def mapping_with_new_keyframe(self, curr_kfm: Keyframe):
        """
        - Removes redundant frames
        - Creates new!! landmarks create_new_landmarks()
        - updates keyframe
        """
        self.curr_kf = curr_kfm
        print("mapping_with_new_keyframe")
        # // set the origin keyframe -> whatever that means?
        # local_map_cleaner_->set_origin_keyframe_id(map_db_->origin_keyfrm_->id_);
        # self.curr_kf = keyframe
        # // store the new keyframe to the database
        # store_new_keyframe();
        self.store_new_keyframe()

        # // remove redundant landmarks
        # local_map_cleaner_->remove_redundant_landmarks(cur_keyfrm_->id_);
        self.remove_redundant_landmarks()

        self.create_new_landmarks(self.data)

    def create_new_landmarks(self, data):
        num_covisibilites = 10
        #TODO: get top n covisibilites
        curr_cam_center = self.curr_kf.world_pose.get_origin()
        covisibilites = []
        print("create_new_landmarks self.keyframes: ", self.covisibility)
        print("create_new_landmarks im_name: ", self.curr_kf.im_name)
        for neighbor_kfm in self.covisibility:
            print(neighbor_kfm)
            if neighbor_kfm == self.curr_kf.im_name:
                continue
            print("create_new_landmarks neighbor_kfm: ", neighbor_kfm)
            n_kfm = self.graph.nodes[neighbor_kfm]['data']
            print(n_kfm)
            # neighbor_kfm = cv
            kf_cam_center = n_kfm.world_pose.get_origin()
            baseline = kf_cam_center - curr_cam_center
            dist = np.linalg.norm(baseline)
            #if monocular
            median_depth = n_kfm.\
                compute_median_depth(True, self.graph, self.reconstruction)
            if dist < 0.02 * median_depth:
                continue

            # TODO: Essential solver between two frames

            #match the top 10 frames!
            matches = self.slam_matcher.\
                match_for_triangulation(self.curr_kf, neighbor_kfm,
                                        self.graph, self.data)
            n_kfm = self.graph.nodes[neighbor_kfm]['data']
            self.triangulate_with_two_kfs(n_kfm, self.curr_kf, matches, data)
        return True
    
    def triangulate_with_two_kfs(self, kf1: Keyframe, kf2: Keyframe, matches, data):

        """kf1 -> neighbor (old) kf, kf2 -> current kf
        """
        #load the features to be triangulated
        #TODO: Think about frame1/2 and matches
        frame1, frame2 = kf1.im_name, kf2.im_name
        p1, f1, c1 = data.load_features(frame1)
        p2, f2, c2 = data.load_features(frame2)
        # Remove already existing landmarks?
        # Get all the landmarks seen in the current kf
        seen_landmarks = self.graph[frame2]
        for lm_id in seen_landmarks:
            e = self.graph.get_edge_data(frame1, lm_id)
            if e is not None:
                # deactivate feature
                p1[e['feature_id'], :] = np.NaN
                # print("p2[e['feature_id'], :]:", p2[e['feature_id'], :])
        # print("p1: ", p1, " matches: ", matches)
        print("len(p1): {}, matches: {}: ".format(len(p1), len(matches)))
        # N ow select the actual matches
        p1 = p1[matches[:, 0]]
        p2 = p2[matches[:, 1]]
        # match
        print("len(p1): {}, len(p2): {} ".format(len(p1), len(p2)))
        # Now, build up the graph for the triangulation

        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(frame1), bipartite=0)
        tracks_graph.add_node(str(frame2), bipartite=0)

        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            x, y, s = p1[track_id, :-1]
            if np.isnan(x):
                continue
            r, g, b = c1[track_id, :]

            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(frame1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))
            x, y, s = p2[track_id, :-1]
            r, g, b = c2[track_id, :]
            tracks_graph.add_edge(str(frame2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))

        cameras = data.load_camera_models()
        camera = next(iter(cameras.values()))
        # print("tri: kf1.worldPose before ", frame1, ": ", kf1.world_pose.rotation,
        #       kf1.world_pose.translation)
        # print("tri: kf2.worldPose before ", frame2, ": ", kf2.world_pose.rotation,
        #       kf2.world_pose.translation)
        rec_tri = types.Reconstruction()
        rec_tri.reference = data.load_reference()
        rec_tri.cameras = cameras

        shot1 = types.Shot()
        shot1.id = frame1
        # print("camera: ", camera)
        shot1.camera = camera
        shot1.pose = kf1.world_pose
        shot1.metadata = reconstruction.get_image_metadata(data, frame1)
        rec_tri.add_shot(shot1)

        shot2 = types.Shot()
        shot2.id = frame2
        shot2.camera = camera
        shot2.pose = kf2.world_pose
        shot2.metadata = reconstruction.get_image_metadata(data, frame2)
        rec_tri.add_shot(shot2)

        graph_inliers = nx.Graph()

        print("Running triangulate shot features")
        np_before = len(rec_tri.points)
        reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
                                                 rec_tri, frame1,
                                                 data.config)
        np_after = len(rec_tri.points)
        print("len(graph_inliers.nodes()): ", len(graph_inliers.nodes()))
        # print("tri: kf1.worldPose after ", frame1, ": ",
        #       kf1.world_pose.rotation, kf1.world_pose.translation)
        # print("tri: kf2.worldPose after ", frame2, ": ",
        #       kf2.world_pose.rotation, kf2.world_pose.translation)
        print("Points before: {} and {} ".format(np_before, np_after))
        # visualize landmarks 2D points in KF <-> 2D points in new KF
        # and also reprojections!
        # draw triangulate features in im1
        # get observations
        edges1 = graph_inliers.edges(frame1)
        edges2 = graph_inliers.edges(frame2)
        print("edges1: ", edges1)
        print("edges2: ", edges2)
        logger.setLevel(logging.INFO)
        points = rec_tri.points
        points3D = np.zeros((len(points), 3))
        for idx, pt3D in enumerate(points.values()):
            points3D[idx, :] = pt3D.coordinates
        DO_VISUALIZE = False
        if DO_VISUALIZE:
            obs1 = []
            for u, v in edges1:
                obs1.append(graph_inliers.get_edge_data(u, v)['feature'])
            print("obs1: ", obs1)
            slam_debug.draw_observations_in_image(np.asarray(obs1), frame1, data, False)
            obs2 = []
            for u, v in edges2:
                obs2.append(graph_inliers.get_edge_data(u, v)['feature'])
            print("obs2: ", obs2)
            slam_debug.draw_observations_in_image(np.asarray(obs2), frame2, data, False)
            logger.setLevel(logging.INFO)
            # exit()
            # draw triangulate features in im2
            # reproject estimated features
            

            # TODO: handle repetetive landmarks
            # visualize
            slam_debug.reproject_landmarks(points3D, np.asarray(obs2),
                                        kf2.world_pose, kf2.im_name,
                                        camera, data, True)

        # Due to some sorting issues, we have to go through
        # graph_inliers by "frames" first
        for _, gi_lm_id in graph_inliers.edges(frame1):
            lm_id = str(self.current_lm_id)
            # print(frame1, frame2, gi_lm_id, lm_id)
            # print("tri lm_id: ", lm_id)
            lm = Landmark(lm_id)
            self.n_landmarks += 1
            self.current_lm_id += 1
            # This is essentially the same as adding it to the graph
            self.add_landmark(lm)
            # print("Added lm: ", lm.lm_id, gi_lm_id)
            # print("KF id: ", frame1, frame2)
            # print("frame1: ", frame1, " gi_lm_id: ", gi_lm_id)
            # print("frame2: ", frame2, " gi_lm_id: ", gi_lm_id)
            # TODO: observations
            self.graph.add_edges_from([(frame1, str(lm_id), graph_inliers.
                                        get_edge_data(frame1, gi_lm_id))])
            self.graph.add_edges_from([(frame2, str(lm_id), graph_inliers.
                                        get_edge_data(frame2, gi_lm_id))])
            lm.compute_descriptor(self.graph)
            lm.update_normal_and_depth(pt3D.coordinates, self.graph)
            # We also have to add the points to the reconstruction

            point = types.Point()
            point.id = str(lm_id)
            point.coordinates = rec_tri.points[gi_lm_id].coordinates
            # print("Adding point {} to reconstruction".format(lm_id))
            self.reconstruction.add_point(point)
            self.local_landmarks.append(lm.lm_id)



        # exit()
        #visualize the new landmarks kf X -> newest kf
        

        
        # Add the landmarks to the graph
        # for (m1,m2) in matches:
        #     #if triangulate

        #     lm = Landmark(self.n_landmarks)
        #     lm.add_observations(frame1, m1)
        #     lm.add_observations(frame2, m2)
            
        #     frame1.add_landmark(lm, m1)
        #     frame2.add_landmark(lm, m2)
        # pass

    def remove_redundant_landmarks(self):
        observed_ratio_thr = 0.3
        num_reliable_keyfrms = 2
        num_obs_thr = 2 #is_monocular_ ? 2 : 3
        state_not_clear = 0
        state_valid = 1
        state_invalid = 2
        lm_state = state_not_clear
        fresh_landmarks = []
        num_removed = 0
        cleaned_landmarks = []
        for lm in fresh_landmarks:
            # if lm.will_be_erased():
            # else:
            if lm.get_observed_ratio() < observed_ratio_thr:
                # if `lm` is not reliable
                # remove `lm` from the buffer and the database
                lm_state = state_invalid
            elif num_reliable_keyfrms + lm.first_kf_id <= self.curr_kf.kf_id \
                    and len(lm.observations) <= num_obs_thr:
                # if the number of the observers of `lm` is small after some
                # keyframes were inserted
                # remove `lm` from the buffer and the database
                lm_state = state_invalid
            elif num_reliable_keyfrms + 1 + lm.first_kf_id <= self.curr_kf.kf_id:
                # if the number of the observers of `lm` is small after some
                # keyframes were inserted
                # remove `lm` from the buffer and the database
                lm_state = state_valid

            if lm_state == state_invalid:
                lm.prepare_for_erasing()
                num_removed += 1
            elif lm_state == state_valid:
                lm.prepare_for_erasing()
            else:
                cleaned_landmarks.append(lm)
                pass
        fresh_landmarks = cleaned_landmarks

    # def determine(self, lm):
    #     """
    #     part of remove_redundant_landmarks
    #     """
    #     # if lm.will_be_erased():
    #     # else:
    #     if lm.get_observed_ratio() < observed_ratio_thr:
    #         # if `lm` is not reliable
    #         # remove `lm` from the buffer and the database
    #         return False
    #     if num_reliable_keyfrms + lm.first_kf_id <= self.curr_kf.kf_id \
    #         and len(lm.observations) <= num_obs_thr:
    #         # if the number of the observers of `lm` is small after some
    #         # keyframes were inserted
    #         # remove `lm` from the buffer and the database
    #         return False
    #     if num_reliable_keyfrms + 1 + lm.first_kf_id <= self.curr_kf.kf_id:
    #         # if the number of the observers of `lm` is small after some
    #         # keyframes were inserted
    #         # remove `lm` from the buffer and the database
    #         return True

    def store_new_keyframe(self):
        curr_lms = self.curr_kf.landmarks_
        print("self.curr_kf: ", self.curr_kf.im_name)
        # asdfasdf
        for lm_id in curr_lms:
            # lm_node = self.graph.node[lm_id]
            # lm: Landmark = lm_node['data']
            #     lm_node = self.graph.node[lm_id]
            lm: Landmark = self.graph.node[lm_id]['data']
            observations = self.graph[lm_id]
            # print("1", self.curr_kf.im_name in observations, self.curr_kf.im_name in observations.keys())
            if self.curr_kf.im_name in observations:
                # TODO: map_cleaner.add_fresh_landmark()
                print("TODO: add_fresh_landmarks()")
            else:
                # print("lm: ", len(self.graph[lm_id]))
                self.graph.add_edge(self.curr_kf.im_name,
                                    lm.lm_id)
                # lm.observations[self.curr_kf.kf_idid] = lm_id
                pos_w = self.reconstruction.points[lm_id].coordinates
                # print("pos_w: ", pos_w)
                lm.update_normal_and_depth(pos_w, self.graph)
                lm.compute_descriptor(self.graph)
        
        #TODO: update graph connections
        #TODO: self.add_keyframe_to_map(self.curr_kf)
        # Is that necessary

    def add_fresh_landmark(self, lm: Landmark):
        self.fresh_landmarks.append(lm)

    # OpenVSlam optimize_current_frame_with_local_map
    def track_with_local_map(self, frame: Frame, slam_tracker):
        """Refine the pose of the current frame with the "local" KFs"""
        print("track_with_local_map", len(frame.landmarks_))
        # acquire more 2D-3D matches by reprojecting the local landmarks to the current frame
        matches = self.search_local_landmarks(frame)
        matches = np.array(matches)
        print("track_with_local_map: matches: ", len(matches))
        observations, _, _ = self.data.load_features(frame.im_name)
        
        print("observations.shape: ", np.shape(observations), matches[:, 0].shape)
        print("observations: ", observations)
        observations = observations[matches[:, 0], 0:3]
        print("observations: ", observations)
        # exit()
        print("len(observations): ", len(observations), observations.shape,
              len(self.local_landmarks))

        points3D = np.zeros((len(observations), 3))
        # for pt in self.reconstruction.points:
        #     print("pt: ", pt)

        print("self.reconstruction: ", len(self.reconstruction.points),
              len(points3D), len(frame.landmarks_), len(matches))
        # generate 3D points
        # for (pt_id, lm_id) in enumerate(frame.landmarks_):
        for (pt_id, (m1, m2)) in enumerate(matches):
            # print("m1 {}, m2 {}".format(m1, m2))
            lm_id = self.local_landmarks[m2]  # frame.landmarks_[m2]
            # lm_id = frame.landmarks_[m2]
            # print("track lm_id ", lm_id)
            print("point3D: ", self.reconstruction.points[str(lm_id)].coordinates)
            points3D[pt_id, :] = \
                self.reconstruction.points[str(lm_id)].coordinates

        print("points3D.shape: ", points3D.shape,
              "observations.shape: ", observations.shape)
        print("frame.world_pose: ", frame.im_name,
              frame.world_pose.rotation, frame.world_pose.translation)
        slam_debug.reproject_landmarks(points3D, observations,
                                       frame.world_pose,
                                       frame.im_name, self.camera[1],
                                       self.data, True)
        # slam_debug.reproject_landmarks(points3D, observations,
        #                                frame.world_pose.inverse(),
        #                                frame.im_name, self.camera[1],
        #                                self.data, True)
        # exit()
        pose, valid_pts = slam_tracker.\
            bundle_tracking(points3D, observations,
                            frame.world_pose, self.camera,
                            self.data.config, self.data)
        
        print("pose after! ", pose.rotation, pose.translation)
        print("valid_pts: ", len(valid_pts))

        # exit()
        
        # for m1, m2 in matches:
        # frame.landmarks_ = self.local_landmarks[matches[:, 1]]
        frame.landmarks_ = self.local_landmarks.copy()
        print("f1: ", len(frame.landmarks_))
        frame.update_visible_landmarks(matches[:, 1])
        print("f2: ", len(frame.landmarks_))
        # frame.update_visible_landmarks_bool(valid_pts)
        frame.landmarks_ = list(compress(frame.landmarks_, valid_pts))
        print("f3: ", len(frame.landmarks_))
        self.num_tracked_lms = len(frame.landmarks_)
        frame.world_pose = pose
        
        #filter outliers and count tracked lms
        # for lm in frame:
        #     lm.num_observable += 1
        #     num_tracked_lms += 1
        #filter outliers
        # if num_tracked_lms < self.num_tracked_lms_thr:
        #     print("Num tracked lms too low %d < %d".
        #           format(num_tracked_lms, self.num_tracked_lms_thr))
        #     return False

        return pose

    # def new_kf_needed(self, num_tracked_lms, frame: Frame):
    def new_kf_needed(self, frame: Frame):
        """Return true if a new keyframe is needed based on the OpenVSLAM criteria
        """
        print("self.n_keyframes: ", self.n_keyframes)
        # Count the number of 3D points observed from more than 3 viewpoints
        min_obs_thr = 3 if 3 <= self.n_keyframes else 2

        # #essentially the graph
        # #find the graph connections
        # #it's about the observations in all frames and not just the kfs
        # #so we can't use the graph of only kfs
        # num_reliable_lms = get_tracked_landmarks(min_obs_thr)
        num_reliable_lms = self.curr_kf.\
            get_num_tracked_landmarks(min_obs_thr, self.graph)
        print("num_reliable_lms: ", num_reliable_lms)
        max_num_frms_ = 30  # the fps
        min_num_frms_ = 0
        
        frm_id_of_last_keyfrm_ = self.curr_kf.kf_id
        print("curr_kf: ", self.curr_kf.kf_id, self.curr_kf.frame_id)
        print("frame.frame_id: ", frame.frame_id, frm_id_of_last_keyfrm_)
        # frame.id
        # ## mapping: Whether is processing
        # #const bool mapper_is_idle = mapper_->get_keyframe_acceptability();
        # Condition A1: Add keyframes if max_num_frames_ or more have passed

        # since the last keyframe insertion
        cond_a1 = (frm_id_of_last_keyfrm_ + max_num_frms_ <= frame.frame_id)
        # Condition A2: Add keyframe if min_num_frames_ or more has passed
        # and mapping module is in standby state
        cond_a2 = (frm_id_of_last_keyfrm_ + min_num_frms_ <= frame.frame_id)
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

        
        # # Do not add if B is not satisfied
        if not cond_b:
            print("not cond_b -> no kf")
            return False

        # # Do not add if none of A is satisfied
        if not cond_a1 and not cond_a2 and not cond_a3:
            print("not cond_a1 and not cond_a2 and not cond_a3 -> no kf")
            return False
        print("NEW KF")
        # exit()
        return True
