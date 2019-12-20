from opensfm import types
from opensfm import reconstruction
from opensfm import feature_loader
from opensfm import csfm
from opensfm.reconstruction import Chronometer
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
        self.feature_ids_last_frame = None
        self.n_tracks = 0
        self.graph = nx.Graph()
        self.reconstruction = []
        self.n_landmarks = 0  # == unique lm id
        self.n_keyframes = 0  # == unique kf id
        self.n_frames = 0     # == unique frame id
        self.curr_kf = None
        # dict because frames can be deleted
        self.keyframes = []     # holds the id, Frame()
        self.frames = {}        # holds the id, Frame()
        # local map keyframes are the ones that have shared
        # landmarks with the current frame
        self.local_keyframes = []
        self.local_landmarks = []
        self.slam_matcher = SlamMatcher(config)
        self.covisibility = nx.Graph()
        self.covisibility_list = []
        self.fresh_landmarks = []
        self.current_lm_i = 0

    def estimate_pose(self):
        # if self.curr_kf is not None:
            # return self.curr_kf.world_pose
        return self.last_frame.world_pose
        # return types.Pose()

    def create_init_map(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
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
            lm.first_kf_id = self.init_frame.kf_id
            self.graph.add_node(lm_id, data=lm)

            int_id = int(lm_id)
            if int_id > max_lm:
                max_lm = int_id
            lm.compute_descriptor(self.graph)
            pos_w = rec_init.points[str(lm_id)].coordinates
            lm.update_normal_and_depth(pos_w, self.graph)
            self.local_landmarks.append(lm_id)
        print("create_init_map: len(local_landmarks): ", self.local_landmarks)
        self.current_lm_id = max_lm

        # also copy them to current kf
        curr_kf.landmarks_ = self.local_landmarks.copy()
        self.init_frame.landmarks_ = self.local_landmarks.copy()
        curr_frame.landmarks_ = self.local_landmarks.copy()

        # go through the graph
        for lm_id in graph_inliers[init_frame.im_name]:
            #get the feature ids
            f1 = self.graph.get_edge_data(init_frame.im_name, lm_id)['feature_id']
            f2 = self.graph.get_edge_data(curr_frame.im_name, lm_id)['feature_id']
            self.init_frame.matched_lms[f1] = lm_id
            curr_kf.matched_lms[f2] = lm_id
            

        # copy local landmarks to last_frame
        self.last_frame.landmarks_ = curr_kf.landmarks_.copy()
        self.last_frame.im_name = curr_kf.im_name

        print("create_init_map with landmarks: ", len(curr_kf.landmarks_),
              len(self.last_frame.landmarks_), len(self.local_landmarks))
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
        self.covisibility_list.append(str(kf.im_name))
        self.n_keyframes += 1
        shot1 = types.Shot()
        shot1.id = kf.im_name
        shot1.camera = self.camera[1]
        print("kf.im_name: ", kf.im_name, "camera: ", self.camera)
        shot1.pose = kf.world_pose
        shot1.metadata = reconstruction.\
            get_image_metadata(self.data, kf.im_name)
        self.reconstruction.add_shot(shot1)
        self.keyframes.append(kf.im_name)

    def add_landmark(self, lm: Landmark):
        """Add landmark to graph"""
        self.graph.add_node(str(lm.lm_id), bipartite=1, data=lm)

    def erase_keyframe(self,  kf: Keyframe):
        self.graph.remove_node(kf.im_name)
        self.covisibility.remove_node(kf.im_name)

    def erase_landmark(self, lm: Landmark):
        self.graph.remove_node(lm.lm_id)

    def fuse_duplicated_landmarks(self):
        print("self.local_keyframes", self.local_keyframes)
        duplicates = 0
        n_original = 0
        for kf_id in self.local_keyframes:
            # read all the landmarks attached to this keyframe
            landmarks = self.graph[kf_id]
            feature_ids = {}
            for lm_id in landmarks:
                edge = self.graph.get_edge_data(kf_id, lm_id)
                feature_id = edge['feature_id']
                elem = feature_ids.get(feature_id)
                if elem is not None:
                    duplicates += 1
                    print("Found duplicate at ", elem, " for ", lm_id)
                    print("elem: ", elem, self.graph[elem])
                    print("lm_id: ", lm_id, self.graph[lm_id])
                    print("edge_elem: ", self.graph.get_edge_data(kf_id, elem))
                    print("edge_lm_id: ", edge)
                    exit()
                else:
                    feature_ids[feature_id] = lm_id
                    n_original += 1
        # create a dict with feature ids
        print("duplicates found: ", duplicates, n_original - duplicates)

        # OpenVSlam style
        # reproject the landmarks observed in the current keyframe to each of the targets, and acquire
        # - additional matches
        # - duplication of matches
        # then, add matches and solve duplication

        # reproject the landmarks observed in each of the targets to each of the current frame, and acquire
        # - additional matches
        # - duplication of matches
        # then, add matches and solve duplication
        pass

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
        print("frame.landmarks_", len(frame.landmarks_))
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
        print("self.local_landmarks: ",
              len(self.local_landmarks), len(self.local_keyframes))
        # count the number of lmid
        lm_count = defaultdict(int)
        for lm in self.local_landmarks:
            lm_count[lm] += 1
        if len(lm_count) > 0:
            print("lm_count", max(lm_count.values()), len(lm_count))

    def apply_landmark_replace(self):
        print('apply landmark?')
        pass

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

    def search_local_landmarks_in_kf(self, frame: Keyframe):
        """Acquire more 2D-3D matches by reprojecting the 
        local landmarks to the current frame

        Return:
            - matches: Nx2 matrix with the [feature_id, landmark_id]
        """
        margin = 5
        print("self.local_landmarks: ", len(self.local_landmarks))
        if len(self.local_landmarks) == 0:
            return []
        matches = self.slam_matcher.\
            match_frame_to_landmarks(frame, self.local_landmarks, margin,
                                     self.data, self.graph)
        print("matches: ", len(matches))
        return matches

    # def fix_keyframes(self, n_keyframes, max_kf_opt=5):
    #     """Creates a bool list of KFs that should stay fixed during opt.
    #     """
    #     if n_keyframes == 0 or max_kf_opt < 0:
    #         return []
    #     n_elem_const = max(n_keyframes-max_kf_opt, 1)
    #     return [True]*n_elem_const + [False]*(n_keyframes-n_elem_const)

    def fix_keyframes(self, n_fix_frames=1, max_kf_opt=5):
        kf_dict = {}
        n_keyframes = len(self.keyframes)
        print("self.n_keyframes", self.n_keyframes)
        print("self.keyframes", self.keyframes)
        if n_keyframes == 0 or max_kf_opt < 0:
            return []
        n_elem_const = max(n_keyframes-max_kf_opt, n_fix_frames)
        for kf_c in self.keyframes[0:n_elem_const]:
            kf_dict[kf_c] = True
        for kf_v in self.keyframes[n_elem_const:]:
            kf_dict[kf_v] = False
        return kf_dict

    def local_bundle_adjustment(self):
        """This is very similar to bundle_tracking
        The main difference is that we add a number
        of frames but "fix" the positions of the oldest.
        """
        print("local_bundle_adjustment!")
        # We can easily build the equation system from the reconstruction
        if self.n_keyframes <= 2:
            return
        ba = csfm.BundleAdjuster()
        
        for camera in self.reconstruction.cameras.values():
            # print(camera.projection_type)
            reconstruction._add_camera_to_bundle(ba, camera, constant=True)
        
        # First, create a list of all frames and fix all but the newest N
        kf_constant = self.fix_keyframes(2, 10)
        kf_added = {}
        print("kf_constant: ", kf_constant)
        n_observations = 0
        n_obs_rel = 0
        for lm_id in self.local_landmarks:
            lm_node = self.graph[lm_id]

            if (len(lm_node) > 2):
                n_obs_rel += len(lm_node) 
            # if (len(lm_node) <= 2):
                # continue
            # add lm
            # lm_id = point_id
            point = self.reconstruction.points[lm_id]
            ba.add_point(point.id, point.coordinates, False)    
            for kf_id in lm_node.keys():
                k = kf_added.get(str(kf_id))
                # add kf if not added
                if k is None:
                    kf_added[str(kf_id)] = True
                    shot = self.reconstruction.shots[kf_id]
                    r = shot.pose.rotation
                    t = shot.pose.translation
                    # print("kf_id", kf_id)
                    ba.add_shot(shot.id, shot.camera.id,
                                r, t, kf_constant[str(kf_id)])
                # add observation
                scale = self.graph[kf_id][lm_id]['feature_scale']
                pt = self.graph[kf_id][lm_id]['feature']
                ba.add_point_projection_observation(
                    kf_id, lm_id, pt[0], pt[1], scale)
                n_observations += 1
        print("kf_added: ", kf_added)
        print("n_observations: ", n_observations, " reliable: ", n_obs_rel)
        config = self.data.config

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
        ba.set_linear_solver_type("DENSE_SCHUR")
        chrono = Chronometer()
        print("Starting local BA")
        chrono.lap('setup')
        ba.run()
        chrono.lap('run')
        print("End local BA")
        print("kf_added: ", kf_added)
        # update frame poses!
        for kf_id in kf_added.keys():
            # don't update fixed ones
            if not kf_constant[kf_id]:
                # update reconstruction
                shot = self.reconstruction.shots[kf_id]
                s = ba.get_shot(shot.id)
                shot.pose.rotation = [s.r[0], s.r[1], s.r[2]]
                shot.pose.translation = [s.t[0], s.t[1], s.t[2]]
                # update kf
                
                kf: Keyframe = self.graph.node[kf_id]['data']
                print("Before: ", kf.im_name, kf.world_pose.rotation,
                     kf.world_pose.translation)
                kf.world_pose.rotation = [s.r[0], s.r[1], s.r[2]]
                kf.world_pose.translation = [s.t[0], s.t[1], s.t[2]]
                print("Updating: ", kf.im_name, kf.world_pose.rotation,
                     kf.world_pose.translation)

        for lm_id in self.local_landmarks:
            point = self.reconstruction.points[lm_id]
            p = ba.get_point(point.id)
            point.coordinates = [p.p[0], p.p[1], p.p[2]]
            point.reprojection_errors = p.reprojection_errors
            print("p.reprojection_errors", lm_id, ":", p.reprojection_errors)

        chrono.lap('teardown')

        logger.debug(ba.brief_report())
        report = {
            'wall_times': dict(chrono.lap_times()),
            'brief_report': ba.brief_report(),
        }
        print("Report: ", report)
        return report

    def search_local_landmarks(self, frame: Frame):
        """ Acquire more 2D-3D matches by reprojecting the 
        local landmarks to the current frame.
        """
        print("search_local_landmarks: ", len(frame.landmarks_))
        # for lm_id in frame.landmarks_:
        #     lm = self.graph.node[lm_id]['data']
        #     lm.is_observable_in_tracking = False
        #     lm.identifier_in_local_lm_search_ = \
        #         frame.frame_id
        #     lm.num_observable += 1

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
        print("frame.landmarks_: bef", len(frame.landmarks_))
        print("self.local_landmarks: ", len(self.local_landmarks))
        frame.landmarks_[:] = list(set(frame.landmarks_+self.local_landmarks))
        print("frame.landmarks_ aft: ", len(frame.landmarks_))
        matches = self.slam_matcher.\
            match_frame_to_landmarks(frame, frame.landmarks_, margin,
                                     self.data, self.graph)
        print("len(matches): ", len(matches))
            # match_frame_to_landmarks(frame, self.local_landmarks, margin,
                                    #  self.data, self.graph)
        # Let's assume that matches are mostly correct and matched landmarks are visible!
        if matches is None:
            return None
        for _, flm_id in matches:
            lm_id = frame.landmarks_[flm_id]
            lm = self.graph.node[lm_id]['data']
            lm.num_observable += 1
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
        if self.curr_kf is not None:
            old_frame = self.curr_kf.im_name
        else:
            old_frame = ""
        self.curr_kf = curr_kfm
        print("mapping_with_new_keyframe", curr_kfm.im_name,
              ", ", old_frame, self.curr_kf.im_name)



        # Store the landmarks seen in frame 2
        in_graph = {}
        frame1 = self.curr_kf.im_name
        seen_landmarks = self.graph[frame1]
        print("frame1: ", frame1, len(seen_landmarks))
        
        for lm_id in seen_landmarks:
            e = self.graph.get_edge_data(frame1, lm_id)
            self.curr_kf.matched_lms[e['feature_id']] = lm_id
            if e['feature_id'] in in_graph:
                print("e(", frame1, ",", lm_id, "): ", e)
                print("Already in there mapping before store!", e['feature_id'], "lm_id: ", lm_id)
                exit()
            in_graph[e['feature_id']] = lm_id

        # // set the origin keyframe -> whatever that means?
        # local_map_cleaner_->set_origin_keyframe_id(map_db_->origin_keyfrm_->id_);
        # self.curr_kf = keyframe
        # // store the new keyframe to the database
        # if (self.init_frame.im_name != self.curr_kf.im_name):
        #     slam_debug.visualize_graph(self.graph, old_frame,
        #                                self.curr_kf.im_name, self.data, False)
        self.store_new_keyframe()

        # Store the landmarks seen in frame 2
        in_graph = {}
        frame1 = self.curr_kf.im_name
        seen_landmarks = self.graph[frame1]
        print("frame1: ", frame1)
        for lm_id in seen_landmarks:
            e = self.graph.get_edge_data(frame1, lm_id)
            # print("mwnk e(", frame1, ",", lm_id, "): ", e)
            if e['feature_id'] in in_graph:
                print("Already in there mapping after store!", e['feature_id'],
                      "lm_id: ", lm_id)
                exit()
            in_graph[e['feature_id']] = lm_id

        print("before: ", self.graph.nodes())
        # // remove redundant landmarks
        # local_map_cleaner_->remove_redundant_landmarks(cur_keyfrm_->id_);
        self.remove_redundant_landmarks()
        print("after: ", self.graph.nodes())
        # TODO: Prevent create new observations from inserting 
        # deleted lms!
        self.create_new_observations_for_lm(self.data)
        self.create_new_landmarks(self.data)
        # if (self.init_frame.im_name != self.curr_kf.im_name):
        #     slam_debug.visualize_graph(self.graph, old_frame,
        #                                self.curr_kf.im_name, self.data, False)
        self.fuse_duplicated_landmarks()

    def create_new_observations_for_lm(self, data):
        print("create_new_observations_for_lm: ", self.curr_kf.im_name)
        print("len(local_landmarks): ", len(self.local_landmarks))
        # first match all the local landmarks to the featues in self.curr_kf
        matches_lm_f = self.search_local_landmarks_in_kf(self.curr_kf)
        p1, _, c1 = feature_loader.instance.load_points_features_colors(
                     data, self.curr_kf.im_name, masked=True)
        print("Want to add: ", len(matches_lm_f))
        n_added = 0
        for (f1_id, loc_lm_id) in matches_lm_f:
            lm_id = self.local_landmarks[loc_lm_id]
            if self.curr_kf.matched_lms[f1_id] != -1:
                # print("Prevent adding edge", self.curr_kf.im_name,
                #      f1_id, " clm_id: ", lm_id, self.graph[lm_id])
                continue
            
            lm: Landmark = self.graph.node[lm_id]['data']
            # print("self.graph.get_edge_data(self.curr_kf.im_name, lm_id)): ",
            #       self.graph.get_edge_data(self.curr_kf.im_name, lm_id))
            # print("Adding edge new obs: ", self.curr_kf.im_name, " clm_id: ", lm_id, "f1_id: ", f1_id)
            x, y, s = p1[f1_id, 0:3]
            r, g, b = c1[f1_id, :]

            # print("Already in graph? create",
                #   self.graph.get_edge_data(self.curr_kf.im_name, lm_id))
            if self.graph.has_node(lm_id):
                # add observations
                self.graph.add_edge(self.curr_kf.im_name, lm_id,
                                    feature=(float(x), float(y)),
                                    feature_scale=float(s),
                                    feature_id=int(f1_id),
                                    feature_color=(float(r), float(g), float(b)))
                pos_w = self.reconstruction.points[lm_id].coordinates
                lm.update_normal_and_depth(pos_w, self.graph)
                lm.compute_descriptor(self.graph)
                self.curr_kf.matched_lms[f1_id] = lm_id
                n_added += 1
                print("Add observation for ", lm_id)
        print("added {} new observations to graph for {} ".
              format(n_added, self.curr_kf.im_name))

    def create_new_landmarks(self, data):
        num_covisibilites = 10
        #TODO: get top n covisibilites
        curr_cam_center = self.curr_kf.world_pose.get_origin()
        # covisibilites = []
        # print("create_new_landmarks self.keyframes: ", self.covisibility)
        # print("create_new_landmarks im_name: ", self.curr_kf.im_name)
        # print("len(local_landmarks): ", len(self.local_landmarks))
        # # first match all the local landmarks to the featues in self.curr_kf
        # matches_lm_f = self.search_local_landmarks_in_kf(self.curr_kf)
        # p1, _, c1 = feature_loader.instance.load_points_features_colors(
        #              data, self.curr_kf.im_name, masked=True)
        # print("Want to add: ", len(matches_lm_f))
        # n_added = 0
        # for (f1_id, loc_lm_id) in matches_lm_f:
        #     lm_id = self.local_landmarks[loc_lm_id]
        #     lm: Landmark = self.graph.node[lm_id]['data']
        #     if self.curr_kf.matched_lms[f1_id]:
        #         print("Prevent adding edge", f1_id)
        #         continue
        #     # observations = self.graph[lm_id]
        #     # if self.curr_kf.im_name in observations:
        #         # TODO: map_cleaner.add_fresh_landmark()
        #         # print("TODO: add_fresh_landmarks()")
        #         # pass
        #     # else:
        #     # f1_id = self.feature_ids_last_frame[lm_id]
        #     print("Adding edge: ", self.curr_kf.im_name, lm_id, "f1_id: ", f1_id)
        #     x, y, s = p1[f1_id, 0:3]
        #     r, g, b = c1[f1_id, :]

        #     print("Already in graph? create",
        #           self.graph.get_edge_data(self.curr_kf.im_name, lm_id))
        #     # e = self.graph.get_edge_data(self.curr_kf.im_name, lm_id)
        #     if self.curr_kf.matched_lms[f1_id]:
        #         print("Adding edge that is in graph!!", f1_id, lm_id)
        #         exit()

        #     # print("self.graph[lm_id]: ", self.graph[lm_id])
        #     #TODO: add feature id
        #     self.graph.add_edge(self.curr_kf.im_name, lm_id,
        #                         feature=(float(x), float(y)),
        #                         feature_scale=float(s),
        #                         feature_id=int(f1_id),
        #                         feature_color=(float(r), float(g), float(b)))

        #     pos_w = self.reconstruction.points[lm_id].coordinates
        #     lm.update_normal_and_depth(pos_w, self.graph)
        #     lm.compute_descriptor(self.graph)
        #     self.curr_kf.matched_lms[f1_id] = True
        #     n_added += 1
        # print("added {} new edges to graph for {} ".
        #       format(n_added, self.curr_kf.im_name))



        #If match with landmark, add an observation

        # for neighbor_kfm in self.covisibility:
        cov_frames = self.covisibility_list[-num_covisibilites:]
        for neighbor_kfm in cov_frames:
            if neighbor_kfm == self.curr_kf.im_name:
                continue
            print("Trying to triangulate: ",
                  neighbor_kfm, "<->", self.curr_kf.im_name)
            n_kfm = self.graph.nodes[neighbor_kfm]['data']
            print("create_new_landmarks neighbor_kfm: ", neighbor_kfm, n_kfm)
            kf_cam_center = n_kfm.world_pose.get_origin()
            baseline = kf_cam_center - curr_cam_center
            dist = np.linalg.norm(baseline)
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
            print("n_kfm: ", n_kfm.im_name, neighbor_kfm)

            self.triangulate_with_two_kfs(self.curr_kf, n_kfm, matches, data)
        
        return True
    

    def triangulate_with_two_kfs(self, kf1: Keyframe, kf2: Keyframe, matches, data):
        """kf1 -> current frame
        kf2 -> frame to triangulate with 
        """
        if matches is None:
            return
        frame1, frame2 = kf1.im_name, kf2.im_name
        p1, f1, c1 = feature_loader.instance.load_points_features_colors(
                     data, frame1, masked=True)
        p2, f2, c2 = feature_loader.instance.load_points_features_colors(
                     data, frame2, masked=True)
        print("len(self.local_landmarks):", len(self.local_landmarks))
        #project

        # Remove already matched features
        # for (idx, (m1, m2)) in enumerate(matches):
            
        #     # Check if already matched
        #     if not self.matched_lms:
        #     #If it is a match, just add the observations
        #     #TODO: don't re-add stuff
        #     # print("Testing m1 {} and m2 {} ".format(m1, m2))
        #     # if a is None:
        #         x, y, s = p1[m1, 0:3]
        #         r, g, b = c1[m1, :]
        #         print("----")
        #         print("data1: ", frame1,
        #               self.graph.get_edge_data(str(a), str(frame1)))
        #         print("data2: ", frame2,
        #               self.graph.get_edge_data(str(a), str(frame2)))
        #         print("adding: ", frame1, "lm_id", a, "x,y,s", x, y, s, 
        #               "r,g,b: ", r, g, b, "m1", m1)
        #         # if self.graph.get_edge_data(str(a), str(frame1))
        #         # ['feature_id'] != m1:
        #             # exit(0)
        #         # self.graph.add_edge(str(a), str(frame1),
        #         #                 feature=(float(x), float(y)),
        #         #                 feature_scale=float(s),
        #         #                 feature_id=int(m1),
        #                         # feature_color=(float(r), float(g), float(b)))
        #         # added_obs[idx] = True
        #         n_added += 1
        #     else:
        #         print("Filtering m1, m2: ", m1, m2)
        #         print("m1 {} already in there".format(m1))
        #         added_obs[idx] = False

        # matches = matches[added_obs, :]
        # print("Remove already added observations: ",
            #   len(matches), len(added_obs))
        # if frame1 == "00006.png" or frame2 == "00006.png":
        # if frame1 == "00005.png" or frame2 == "00005.png":
            # exit()

        # print("len(p1): {}, matches: {}: ".format(len(p1), len(matches)))
        
        # N ow select the actual matches
        # p1 = p1[matches[:, 0]]
        # p2 = p2[matches[:, 1]]
        # match
        print("len(p1): {}, len(p2): {} ".format(len(p1), len(p2)))
        print("len(kf1.matched_lms): {}, len(kf2.matched_lms): {} "
              .format(len(kf1.matched_lms), len(kf2.matched_lms)))
        # Now, build up the graph for the triangulation

        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(frame1), bipartite=0)
        tracks_graph.add_node(str(frame2), bipartite=0)

        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            # this checks whether the current kf was matched
            # to one of the landmarks.
            # if f2 is already in a lm
            
            if kf1.matched_lms[f1_id] == -1:
                old_lm_id = kf2.matched_lms[f2_id]
                if old_lm_id != -1:
                    if self.graph.has_node(old_lm_id):
                        # also add the new track
                        print("Not matched in current frame but matched to other frame!", old_lm_id,
                                kf1.matched_lms[f1_id])
                        # if old_lm_id != -1:
                        print("new! triang: track_id  {}, frames: {}<->{} f1_id {}, f2_id {}".
                            format(old_lm_id, frame1, frame2, f1_id, f2_id))
                        x, y, s = p1[f1_id, 0:3]
                        r, g, b = c1[f1_id, :]
                        self.graph.add_edge(str(frame1),
                                            str(old_lm_id),
                                            feature=(float(x), float(y)),
                                            feature_scale=float(s),
                                            feature_id=int(f1_id),
                                            feature_color=(float(r), float(g), float(b)))
                        kf1.matched_lms[f1_id] = old_lm_id
                else:
            # if kf1.matched_lms[f1_id] == -1:
                    # else:
                    print("triang: track_id  {}, frames: {}<->{} f1_id {}, f2_id {}".
                        format(track_id, frame1, frame2, f1_id, f2_id))
                    x, y, s = p2[f2_id, 0:3]
                    r, g, b = c2[f2_id, :]
                    tracks_graph.add_node(str(track_id), bipartite=1)
                    tracks_graph.add_edge(str(frame2),
                                        str(track_id),
                                        feature=(float(x), float(y)),
                                        feature_scale=float(s),
                                        feature_id=int(f2_id),
                                        feature_color=(float(r), float(g), float(b)))

                    x, y, s = p1[f1_id, 0:3]
                    r, g, b = c1[f1_id, :]
                    tracks_graph.add_edge(str(frame1),
                                        str(track_id),
                                        feature=(float(x), float(y)),
                                        feature_scale=float(s),
                                        feature_id=int(f1_id),
                                        feature_color=(float(r), float(g), float(b)))
            # else:
                # print("tracks_graph: f1_id {}, f2_id {} already in graph!".format(f1_id, f2_id))

        cameras = data.load_camera_models()
        camera = next(iter(cameras.values()))
        rec_tri = types.Reconstruction()
        rec_tri.reference = data.load_reference()
        rec_tri.cameras = cameras

        shot1 = types.Shot()
        shot1.id = frame1
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

        print("Running triangulate shot features for ", frame2)
        np_before = len(rec_tri.points)
        reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
                                                 rec_tri, frame1,
                                                 data.config)
        np_after = len(rec_tri.points)
        print("Created len(graph_inliers.nodes()): ",
              len(graph_inliers.nodes()))
        print("Points before: {} and {} ".format(np_before, np_after))
        # visualize landmarks 2D points in KF <-> 2D points in new KF
        # and also reprojections!
        # draw triangulate features in im1
        # get observations
        edges1 = graph_inliers.edges(frame1)
        edges2 = graph_inliers.edges(frame2)
        # we have the edges
        # try to find the same feature already existing in the graph!
        n_duplicates = 0
        for u, v in edges1:
            feature_id = graph_inliers.get_edge_data(u, v)['feature_id']
            for lm_id in self.graph[frame1]:
                feature_id2 = self.graph.\
                    get_edge_data(frame1, lm_id)['feature_id']
                if feature_id == feature_id2:
                    print("created feature already in graph",
                          feature_id, "<->", feature_id2)
                    print("u,v", u, v)
                    print("frame1", frame1, "lm_id", lm_id)
                    print(self.graph[lm_id])
                    n_duplicates += 1
                    exit()
        print("Created landmarks ", np_after, " with ",
              n_duplicates, " duplicates.")
        print("edges1: ", edges1)
        print("edges2: ", edges2)
        points = rec_tri.points
        points3D = np.zeros((len(points), 3))
        for idx, pt3D in enumerate(points.values()):
            points3D[idx, :] = pt3D.coordinates
        # Due to some sorting issues, we have to go through
        matches_dbg = np.zeros([len(graph_inliers.edges(frame1)), 2], dtype=int)
        idx = 0
        # graph_inliers by "frames" first
        for _, gi_lm_id in graph_inliers.edges(frame1):
            lm_id = str(self.current_lm_id)
            lm = Landmark(lm_id)
            lm.first_kf_id = kf1.kf_id
            self.n_landmarks += 1
            self.current_lm_id += 1
            # This is essentially the same as adding it to the graph
            self.add_landmark(lm)
            # Now, relate the gi_lm_id to the actual feature_id
            e1 = graph_inliers.get_edge_data(frame1, gi_lm_id)
            e2 = graph_inliers.get_edge_data(frame2, gi_lm_id)
            
            self.graph.add_edges_from([(frame1, str(lm_id), e1)])
            self.graph.add_edges_from([(frame2, str(lm_id), e2)])

            # also add the observations
            kf1.matched_lms[e1['feature_id']] = lm_id
            kf2.matched_lms[e2['feature_id']] = lm_id
            print("Creating landmarkd: ", lm_id,
                  " between ", frame1, " and ", frame2,
                  e1['feature_id'], "/", e2['feature_id'])
            matches_dbg[idx, :] = np.array([e1['feature_id'], e2['feature_id']])
            idx += 1
            # print("graph_inliers.get_edge_data(",frame1, "gi_lm_id): ", graph_inliers.
                                        # get_edge_data(frame1, gi_lm_id), "lm_id: ", lm_id)
            # print("graph_inliers.get_edge_data(", frame2, "gi_lm_id): ", graph_inliers.
                                        # get_edge_data(frame2, gi_lm_id), "lm_id: ", lm_id)
            lm.compute_descriptor(self.graph)
            lm.update_normal_and_depth(pt3D.coordinates, self.graph)
            # We also have to add the points to the reconstruction
            point = types.Point()
            point.id = str(lm_id)
            point.coordinates = rec_tri.points[gi_lm_id].coordinates
            self.reconstruction.add_point(point)
            self.local_landmarks.append(lm.lm_id)
            self.add_fresh_landmark(lm.lm_id)

        if (len(matches_dbg) > 0):
            print("Newly created landmarks!")
            slam_debug.visualize_matches(matches_dbg, frame1, frame2, data, False)
            points3D_debug = np.zeros([len(rec_tri.points), 3])
            for idx, p in enumerate(rec_tri.points.values()):
                points3D_debug[idx, :] = p.coordinates
            slam_debug.reproject_landmarks(points3D_debug, np.zeros([len(rec_tri.points), 2]),
                                        kf2.world_pose, kf2.im_name, camera,
                                        self.data, title="triangulated: "+kf2.im_name, do_show=True)

    def triangulate_with_two_kfs_old(self, kf1: Keyframe, kf2: Keyframe, matches, data):

        """kf1 -> neighbor (old) kf, kf2 -> current kf
        """
        #load the features to be triangulated
        #TODO: Think about frame1/2 and matches
        frame1, frame2 = kf1.im_name, kf2.im_name
        # p1, f1, c1 = data.load_features(frame1)
        # p2, f2, c2 = data.load_features(frame2)
        p1, f1, c1 = feature_loader.instance.load_points_features_colors(
                     data, frame1, masked=True)
        p2, f2, c2 = feature_loader.instance.load_points_features_colors(
                     data, frame2, masked=True)

        slam_debug.visualize_matches(matches, frame1, frame2, data, True)
        # exit()
        # Maybe we have double matches
        d_m1 = defaultdict(int)
        d_m2 = defaultdict(int)
        for (m1, m2) in matches:
            d_m1[m1] += 1
            d_m2[m2] += 1
            if d_m1[m1] > 1 or d_m2[m2] > 1:
                print("Double matches!!", m1, m2)
                exit()
        in_graph = {}
        # seen_landmarks = self.graph[frame2]
        seen_landmarks = self.graph[frame1]
        print("frame1: ", frame1, " frame2: ", frame2)
        for lm_id in seen_landmarks:
            e = self.graph.get_edge_data(frame1, lm_id)
            if e is None:
                continue
            if e['feature_id'] in in_graph:
                e2 = self.graph.get_edge_data(frame2, lm_id)
                print("e(", frame1, ",", lm_id, "): ", e)
                print("e2(", frame2, ",", lm_id, "): ", e2)
                print("Already in there first!", e['feature_id'],
                      "lm_id: ", lm_id)
                exit()
            in_graph[e['feature_id']] = lm_id

        print("len(in_graph)", len(in_graph),
              "frames: {} {}".format(frame1, frame2))
        added_obs = np.ones(len(matches), dtype=bool)
        n_added = 0
        for (idx, (m1, m2)) in enumerate(matches):
            # if the feature id is matched a is not none
            a = in_graph.get(m1)
            #If it is a match, just add the observations
            #TODO: don't re-add stuff
            print("Testing m1 {} and m2 {} ".format(m1, m2))
            if a is None:
                x, y, s = p1[m1, 0:3]
                r, g, b = c1[m1, :]
                print("----")
                print("data1: ", frame1,
                      self.graph.get_edge_data(str(a), str(frame1)))
                print("data2: ", frame2,
                      self.graph.get_edge_data(str(a), str(frame2)))
                print("adding: ", frame1, "lm_id", a, "x,y,s", x, y, s, 
                      "r,g,b: ", r, g, b, "m1", m1)
                # if self.graph.get_edge_data(str(a), str(frame1))
                # ['feature_id'] != m1:
                    # exit(0)
                # self.graph.add_edge(str(a), str(frame1),
                #                 feature=(float(x), float(y)),
                #                 feature_scale=float(s),
                #                 feature_id=int(m1),
                #                 feature_color=(float(r), float(g), float(b)))
                added_obs[idx] = True
                n_added += 1
            else:
                print("Filtering m1, m2: ", m1, m2)
                print("m1 {} already in there".format(m1))
                added_obs[idx] = False

        print("n_added: ", n_added)

        # seen_landmarks = self.graph[frame2]
        # for lm_id in seen_landmarks:
        #     e = self.graph.get_edge_data(frame2, lm_id)
        #     print("seen_landmarks: frame2: ", frame2, ", ", lm_id)
        #     print("e: ", e)
        #     p2[e['feature_id'], :] = np.NaN
            # e2 = self.graph.get_edge_data(frame2, lm_id)
            # if e2 is not None: 
            # n_frame2_f
                # print("p2[e['feature_id'], :]:", p2[e['feature_id'], :])
        # print("len(seen_landmarks)", len(seen_landmarks), "/", n_frame1_f)
        # print("p1: ", p1, " matches: ", matches)
        print("len(matches)", len(matches), added_obs.shape)
        print("len(matches)", matches.shape, added_obs.shape)
        matches = matches[added_obs, :]
        print("Remove already added observations: ",
              len(matches), len(added_obs))
        # if frame1 == "00006.png" or frame2 == "00006.png":
        # if frame1 == "00005.png" or frame2 == "00005.png":
            # exit()

        # print("len(p1): {}, matches: {}: ".format(len(p1), len(matches)))
        
        # N ow select the actual matches
        # p1 = p1[matches[:, 0]]
        # p2 = p2[matches[:, 1]]
        # match
        print("len(p1): {}, len(p2): {} ".format(len(p1), len(p2)))
        # Now, build up the graph for the triangulation

        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(frame1), bipartite=0)
        tracks_graph.add_node(str(frame2), bipartite=0)

        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            print("track_id {}, frames: {}<->{} f1_id {}, f2_id {}".
                  format(track_id, frame1, frame2, f1_id, f2_id))
            x, y, s = p2[f2_id, 0:3]
            if np.isnan(x):
                continue
            r, g, b = c2[f2_id, :]
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(frame2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))

            x, y, s = p1[f1_id, 0:3]
            r, g, b = c1[f1_id, :]
            tracks_graph.add_edge(str(frame1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))


        # for (track_id, (f1_id, f2_id)) in enumerate(matches):
        #     x, y, s = p1[track_id, :-1]
        #     if np.isnan(x):
        #         continue
        #     r, g, b = c1[track_id, :]

        #     tracks_graph.add_node(str(track_id), bipartite=1)
        #     tracks_graph.add_edge(str(frame1),
        #                           str(track_id),
        #                           feature=(float(x), float(y)),
        #                           feature_scale=float(s),
        #                           feature_id=int(f1_id),
        #                           feature_color=(float(r), float(g), float(b)))
        #     x, y, s = p2[track_id, :-1]
        #     r, g, b = c2[track_id, :]
        #     tracks_graph.add_edge(str(frame2),
        #                           str(track_id),
        #                           feature=(float(x), float(y)),
        #                           feature_scale=float(s),
        #                           feature_id=int(f2_id),
        #                           feature_color=(float(r), float(g), float(b)))

        cameras = data.load_camera_models()
        camera = next(iter(cameras.values()))
        rec_tri = types.Reconstruction()
        rec_tri.reference = data.load_reference()
        rec_tri.cameras = cameras

        shot1 = types.Shot()
        shot1.id = frame1
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

        print("Running triangulate shot features for ", frame2)
        np_before = len(rec_tri.points)
        reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
                                                 rec_tri, frame1,
                                                 data.config)
        np_after = len(rec_tri.points)
        print("Created len(graph_inliers.nodes()): ",
              len(graph_inliers.nodes()))
        print("Points before: {} and {} ".format(np_before, np_after))
        # visualize landmarks 2D points in KF <-> 2D points in new KF
        # and also reprojections!
        # draw triangulate features in im1
        # get observations
        edges1 = graph_inliers.edges(frame1)
        edges2 = graph_inliers.edges(frame2)
        # we have the edges
        # try to find the same feature already existing in the graph!
        n_duplicates = 0
        for u, v in edges1:
            feature_id = graph_inliers.get_edge_data(u, v)['feature_id']
            for lm_id in self.graph[frame1]:
                feature_id2 = self.graph.\
                    get_edge_data(frame1, lm_id)['feature_id']
                if feature_id == feature_id2:
                    print("created feature already in graph",
                          feature_id, "<->", feature_id2)
                    print("u,v", u, v)
                    print("frame1", frame1, "lm_id", lm_id)
                    print(self.graph[lm_id])
                    n_duplicates += 1

        print("Created landmarks ", np_after, " with ",
              n_duplicates, " duplicates.")
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

        # Due to some sorting issues, we have to go through
        # graph_inliers by "frames" first
        for _, gi_lm_id in graph_inliers.edges(frame1):
            lm_id = str(self.current_lm_id)
            lm = Landmark(lm_id)
            self.n_landmarks += 1
            self.current_lm_id += 1
            # This is essentially the same as adding it to the graph
            self.add_landmark(lm)
            # TODO: observations
            self.graph.add_edges_from([(frame1, str(lm_id), graph_inliers.
                                        get_edge_data(frame1, gi_lm_id))])
            self.graph.add_edges_from([(frame2, str(lm_id), graph_inliers.
                                        get_edge_data(frame2, gi_lm_id))])
            
            print("graph_inliers.get_edge_data(",frame1, "gi_lm_id): ", graph_inliers.
                                        get_edge_data(frame1, gi_lm_id), "lm_id: ", lm_id)
            print("graph_inliers.get_edge_data(", frame2, "gi_lm_id): ", graph_inliers.
                                        get_edge_data(frame2, gi_lm_id), "lm_id: ", lm_id)
            lm.compute_descriptor(self.graph)
            lm.update_normal_and_depth(pt3D.coordinates, self.graph)
            # We also have to add the points to the reconstruction
            point = types.Point()
            point.id = str(lm_id)
            point.coordinates = rec_tri.points[gi_lm_id].coordinates
            self.reconstruction.add_point(point)
            self.local_landmarks.append(lm.lm_id)

        points3D_debug = np.zeros([len(rec_tri.points),3])
        for idx, p in enumerate(rec_tri.points.values()):
            # pos_w = self.graph.node[lm.lm_id]
            points3D_debug[idx, :] = p.coordinates
        print("Visualizing ")
        slam_debug.reproject_landmarks(points3D_debug, np.zeros([len(rec_tri.points),2]),
                                       kf2.world_pose, kf2.im_name, camera, self.data, 
                                       title="triangulated: "+kf2.im_name, do_show=True)        

    def remove_redundant_landmarks(self):
        # return
        observed_ratio_thr = 0.3
        num_reliable_keyfrms = 2
        num_obs_thr = 2 #is_monocular_ ? 2 : 3
        lm_not_clear = 0
        lm_valid = 1
        lm_invalid = 2
        # lm_state = lm_not_clear
        # fresh_landmarks = []
        fresh_landmarks = self.fresh_landmarks
        num_removed = 0
        removed_landmarks = []
        cleaned_landmarks = []
        # for lm in fresh_landmarks:
        print("len(fresh_landmarks): ", len(fresh_landmarks))
        for lm_id in fresh_landmarks:
            lm_state = lm_not_clear
            if not self.graph.has_node(lm_id):
                removed_landmarks.append(lm_id)
                continue
                # num_removed

            lm = self.graph.node[lm_id]['data']
            num_observations = len(self.graph[lm_id])
            if num_observations >= 3:
                print("num_observation ", num_observations, " lm_id: ", lm_id)
                # exit()

            # if lm.will_be_erased():
            # else:
            print("lm_id: ", lm_id, lm.get_observed_ratio())
            if lm.get_observed_ratio() < observed_ratio_thr:
                # if `lm` is not reliable
                # remove `lm` from the buffer and the database
                lm_state = lm_invalid
                print("lm {} invalid due to obs_ratio {} < {}".
                      format(lm_id, lm.get_observed_ratio(), observed_ratio_thr))
            elif num_reliable_keyfrms + lm.first_kf_id <= self.curr_kf.kf_id \
                    and num_observations <= num_obs_thr:
                # if the number of the observers of `lm` is small after some
                # keyframes were inserted
                # remove `lm` from the buffer and the database
                lm_state = lm_invalid
                print("lm {} invalid due rel. kfs {} + {} <= {} and {} <= {}".
                      format(lm_id, num_reliable_keyfrms, lm.first_kf_id,
                       self.curr_kf.kf_id, num_observations, num_obs_thr))

            elif num_reliable_keyfrms + 1 + lm.first_kf_id <= self.curr_kf.kf_id:
                # if the number of the observers of `lm` is small after some
                # keyframes were inserted
                # remove `lm` from the buffer and the database
                lm_state = lm_valid
                print("lm {} valid due rel. kfs {} + 1 + {} <= {}".
                      format(lm_id, num_reliable_keyfrms, lm.first_kf_id, self.curr_kf.kf_id))
            # print("lm {} invalid by default".format(lm_id))
            print("lm {} default invalid due rel. kfs {} + 1 + {} <= {}".
                      format(lm_id, num_reliable_keyfrms, lm.first_kf_id, self.curr_kf.kf_id))

            if lm_state == lm_invalid:
                # lm.prepare_for_erasing()
                num_removed += 1
                removed_landmarks.append(lm_id)
            else:
                cleaned_landmarks.append(lm_id)
                # self.graph.remove
            # elif lm_state == state_valid:
                # lm.prepare_for_erasing()
            # else:
                # pass
        # fresh_landmarks = cleaned_landmarks
        for lm_id in removed_landmarks:
            print("remove_landmarkd with id ", lm_id)
            if self.graph.has_node(lm_id):
                print("Remvoe from graph and rec ", lm_id)
                self.graph.remove_node(lm_id)
                
                print("self.graph.has_node(lm_id): ", self.graph.has_node(lm_id))
                print("self.reconstruction.points[{}] = {}".format(lm_id, self.reconstruction.points[lm_id]))
                print("self.reconstruction.points[{}] = {}".format(lm_id, self.reconstruction.points[str(lm_id)]))
                
                del self.reconstruction.points[lm_id]
                

        print("remove_landmark: ", len(removed_landmarks), " cleaned_landmarks: ", len(cleaned_landmarks))

        # somehow also remove from frame.landmarks_
        # clean-up frame.landmarks_
        keep_idx = np.zeros(len(self.curr_kf.landmarks_), dtype=bool)
        for idx, lm_id in enumerate(self.curr_kf.landmarks_):
            if self.graph.has_node(lm_id):
               keep_idx[idx] = True
               print("keep: ", idx, " lm_id: ", lm_id)
            
        self.curr_kf.landmarks_[:] = compress(self.curr_kf.landmarks_, keep_idx)

        keep_idx = np.zeros(len(self.local_landmarks), dtype=bool)
        for idx, lm_id in enumerate(self.local_landmarks):
            if self.graph.has_node(lm_id):
               keep_idx[idx] = True
        
        self.local_landmarks[:] = compress(self.local_landmarks, keep_idx)

        keep_idx = np.zeros(len(self.last_frame.landmarks_), dtype=bool)
        for idx, lm_id in enumerate(self.last_frame.landmarks_):
            if self.graph.has_node(lm_id):
               keep_idx[idx] = True
        
        self.last_frame.landmarks_[:] = compress(self.last_frame.landmarks_, keep_idx)

        print("len: curr_kf: {}, local_lms: {}, landmarks: {}".
              format(len(self.curr_kf.landmarks_),
                     len(self.local_landmarks),
                     len(self.last_frame.landmarks_)))

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
        print("store_new_keyframe kf: ", self.curr_kf.im_name)
        print("store_new_keyframe curr_lms: ", len(curr_lms))

        # feature_ids in last frame
        # p, f, c = self.data.load_features(self.curr_kf.im_name)
        # print("load_features: ", len(p))
        p, f, c = feature_loader.instance.load_points_features_colors(
                     self.data, self.curr_kf.im_name, masked=True)
        print("load_points_features: ", len(p))

        for idx, lm_id in enumerate(curr_lms):
            lm: Landmark = self.graph.node[lm_id]['data']
            observations = self.graph[lm_id]
            if self.curr_kf.im_name in observations:
                # TODO: map_cleaner.add_fresh_landmark()
                # print("TODO: add_fresh_landmarks()")
                # pass
                self.add_fresh_landmark(lm_id)
            else:
                
                f1_id = self.feature_ids_last_frame[lm_id]
                print("Already in graph? store", self.graph.get_edge_data(self.curr_kf.im_name, lm_id))
                if self.curr_kf.matched_lms[f1_id] != -1 or self.graph.get_edge_data(self.curr_kf.im_name, lm_id) is not None:
                    print("Adding an already matched edge!", f1_id, self.curr_kf.im_name, lm_id)
                    exit()
                print(self.curr_kf.im_name, "Adding edge: ", " lm_id: ", lm_id, "f1_id: ", f1_id)
                self.curr_kf.matched_lms[f1_id] = lm_id
                x, y, s = p[f1_id, 0:3]
                r, g, b = c[f1_id, :]


                # print("self.graph[lm_id]: ", self.graph[lm_id])
                #TODO: add feature id
                if self.graph.has_node(lm_id):
                    self.graph.add_edge(self.curr_kf.im_name, lm_id,
                                        feature=(float(x), float(y)),
                                        feature_scale=float(s),
                                        feature_id=int(f1_id),
                                        feature_color=(float(r), float(g), float(b)))

                
                pos_w = self.reconstruction.points[lm_id].coordinates
                lm.update_normal_and_depth(pos_w, self.graph)
                lm.compute_descriptor(self.graph)
        
        #TODO: update graph connections
        #TODO: self.add_keyframe_to_map(self.curr_kf)
        # Is that necessary

    def add_fresh_landmark(self, lm_id):
        # return
        self.fresh_landmarks.append(lm_id)

    def optimize_current_frame_with_local_map(self, frame: Frame, slam_tracker):
        """Refine the pose of the current frame with the "local" KFs
        and add local landmarks to landmarks in frame
        """
        # acquire more 2D-3D matches by reprojecting the local landmarks to the current frame
        # Remove out of bounds landmarks from old frame
        # first add last frames landmarks to current landmarks
        frame.landmarks_ = self.last_frame.landmarks_
        # remove landmarks not in image
        # add landmarks from local landmarks

        # optimize with local map
        # ... Optimize ...

        # Remove outliers after optimization
        




    # OpenVSlam optimize_current_frame_with_local_map
    def track_with_local_map(self, frame: Frame, slam_tracker):
        """Refine the pose of the current frame with the "local" KFs"""
        print("track_with_local_map", len(frame.landmarks_))
        # acquire more 2D-3D matches by reprojecting the local landmarks to the current frame
        matches = self.search_local_landmarks(frame)
        matches = np.array(matches)
        print("track_with_local_map: matches: ", len(matches))
        # observations, _, _ = self.data.load_features(frame.im_name)
        observations, _, _ = \
            feature_loader.instance.load_points_features_colors(
                    self.data, frame.im_name, masked=True)
        print("load_features: ", len(observations))
        print("observations.shape: ", np.shape(observations), matches[:, 0].shape)
        # print("observations: ", observations)
        observations = observations[matches[:, 0], 0:3]
        # print("observations: ", observations)
        print("len(observations): ", len(observations), observations.shape,
              len(self.local_landmarks))

        points3D = np.zeros((len(observations), 3))

        print("self.reconstruction: ", len(self.reconstruction.points),
              len(points3D), len(frame.landmarks_), len(matches))
        # generate 3D points
        # for (pt_id, (m1, m2)) in enumerate(matches):
        #     lm_id = self.local_landmarks[m2]
        #     points3D[pt_id, :] = \
        #         self.reconstruction.points[str(lm_id)].coordinates
        for (pt_id, (m1, m2)) in enumerate(matches):
            lm_id = frame.landmarks_[m2] #self.local_landmarks[m2]
            points3D[pt_id, :] = \
                self.reconstruction.points[str(lm_id)].coordinates


        print("points3D.shape: ", points3D.shape,
              "observations.shape: ", observations.shape)
        print("frame.world_pose: ", frame.im_name,
              frame.world_pose.rotation, frame.world_pose.translation)
        slam_debug.reproject_landmarks(points3D, observations, frame.world_pose, 
                                       frame.im_name, self.camera[1], self.data,
                                       title="bef tracking: "+frame.im_name, do_show=False)
        pose, valid_pts = slam_tracker.\
            bundle_tracking(points3D, observations,
                            frame.world_pose, self.camera,
                            self.data.config, self.data)
        
        print("pose after! ", pose.rotation, pose.translation)
        print("valid_pts: ", len(valid_pts), " vs ", len(observations))
        slam_debug.reproject_landmarks(points3D, observations,
                                       pose, frame.im_name, self.camera[1], self.data,
                                       title="aft tracking: "+frame.im_name, do_show=True)
        
        # frame.landmarks_ = self.local_landmarks.copy()
        frame.update_visible_landmarks(matches[:, 1])
        frame.landmarks_ = list(compress(frame.landmarks_, valid_pts))
        self.num_tracked_lms = len(frame.landmarks_)
        frame.world_pose = pose
        m = matches[:, 0][valid_pts]
        self.feature_ids_last_frame = {}
        # add observations
        for idx, lm_id in enumerate(frame.landmarks_):
            m1 = m[idx]
            self.feature_ids_last_frame[lm_id] = m1
            lm = self.graph.nodes[lm_id]['data']
            lm.num_observed += 1
        return pose

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
        min_num_frms_ = 2
        
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
