from opensfm import types
from opensfm import reconstruction
from slam_types import Frame
from slam_types import Keyframe
from slam_types import Landmark
# from slam_tracker import SlamTracker
from slam_matcher import SlamMatcher
from collections import defaultdict
import networkx as nx
import logging
import numpy as np
logger = logging.getLogger(__name__)


class SlamMapper(object):

    def __init__(self, data, config):
        """SlamMapper holds a local and global map
        """
        self.data = data
        self.reconstruction = None
        self.last_frame = Frame("dummy")
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

    def create_init_map(self, graph_inliers, reconstruction_init,
                        init_frame: Frame, curr_frame: Frame):
        """Basically the graph contains
        the keyframes/shots and landmarks.
        Edges are connections between keyframes
        and landmarks.
        """
        self.graph = graph_inliers
        self.reconstruction = reconstruction_init
        # Create keyframes
        self.init_frame = Keyframe(init_frame, self.data)
        self.init_frame.world_pose = \
            reconstruction_init.shots[init_frame.im_name].pose
        self.curr_kf = Keyframe(curr_frame, self.data)
        self.curr_kf.world_pose = reconstruction_init.shots[curr_frame.im_name].pose
        
        #Add to data and covisibility
        self.add_keyframe(self.init_frame)
        self.add_keyframe(self.curr_kf)

        max_lm = 0
        #Add landmarks to nodes
        for lm_id in self.graph[self.init_frame.im_name]:
            lm = Landmark(int(lm_id))
            self.graph.add_node(lm_id, data=lm)
            int_id = int(lm_id)
            if int_id > max_lm:
                max_lm = int_id
            lm.compute_descriptor(self.graph)
            lm.update_normal_and_depth(self.graph)
        
        self.current_lm_id = max_lm

        SET THE curr_frame WITH THE INIT/KF,....
        # pass

    # def compute_descriptor(self, lm: Landmark):
    #     """Computes the descriptor of the lm
    #     from all the observations
    #     Take random descriptor
    #     """

    #     keyframes = self.graph[lm.lm_id]
    #     # descriptors = []
    #     for kf_name in keyframes:
    #         kf = self.graph.node[kf_name]['data']
    #         track = self.graph[kf_name, lm.lm_id]
    #         lm.descriptor = kf.descriptors[track['feature_id']]
    #         return
        

    def add_keyframe(self, kf: Keyframe):
        """Adds a keyframe to the map graph
        and the covisibility graph
        """
        # self.keyframes[keyframe.]
        self.graph.add_node(str(kf.im_name), bipartitite=0, data=kf)
        self.covisibility.add_node(str(kf.im_name))

    def add_landmark(self, lm: Landmark):
        """Add landmark to graph"""
        self.graph.add_node(str(lm.lm_id), bipartite=1, data=lm)

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

    def set_last_frame(self, frame):
        """Sets the last frame

        Arguments:
            frame: of Frame
        """
        self.n_frames += 1
        self.last_frame = frame
        self.frames[frame.id] = frame

    def add_frame_to_reconstruction(self, frame, pose, camera, data):
        shot1 = types.Shot()
        shot1.id = frame
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

    
    def update_local_keyframes(self):
        """Count number of lm shared between current frame and neighbour KFs
        (count obs.). For each keyframe, we keep count of how many lms it
        shares with the current one.
        """
        print("update_local_keyframes")
        kfs_weights = defaultdict(int)
        print("self.curr_kf: ", self.curr_kf)
        for lm in self.curr_kf.landmarks:
            # find the number of sharing landmarks between 
            # the current frame and each of the neighbor keyframes
            connected_kfs = self.graph[lm]
            for kfs in connected_kfs:
                kfs_weights[kfs.id] += 1
        
        if len(kfs_weights) == 0:
            return
        
        # kfs_weights = sorted(kfs_weights)
        self.local_keyframs.clear()
        max_weight = 0
        nearest_frame = -1
        for kf_id, weight in kfs_weights.items():
            self.local_keyframes.append(kf_id)
            self.keyframes[kf_id].local_map_update_identifier = kf_id
            if weight > max_weight:
                max_weight = weight
                nearest_frame = kf_id
        
        # max_local_keyframes = 60
        # add the second-order keyframes to the local landmarks
        # for local_kf in self.local_keyframes:
        #     if len(self.local_keyframes) > max_local_keyframes:
        #         break

    def update_local_landmarks(self, graph):
        """Update local landmarks by adding
        all the landmarks of the local keyframes.
        """
        print("update_local_landmarks")
        for kf_id in self.local_keyframes:
            # lms = graph[kf_id]
            for lm_node in graph[kf_id]:
                lm = lm_node['data']
                #Avoid duplication
                if lm.local_map_update_identifier == kf_id:
                    continue
                lm.local_map_update_identifier = kf_id
                self.local_landmarks.append(lm)

            # self.keyframes[kf_id]
            
    def apply_landmark_replace(self):
        print('apply landmark?')

    def set_local_landmarks(self):
        print("set_local_landmarks()")
        
    def update_local_map(self):
        # Todo: unify update_local_kf, keyframes and set
        self.update_local_keyframes()
        self.update_local_landmarks(self.graph)
        # self.set_local_landmarks()
        #landmarks are already set
        return True
    
    def search_local_landmarks(self, frame: Frame):
        """ Acquire more 2D-3D matches by reprojecting the 
        local landmarks to the current frame
        """
        for lm in frame.visible_landmarks:
            lm.is_observable_in_tracking = False
            lm.identifier_in_local_lm_search_ = \
                frame.identifier_in_local_lm_search_
            # lm.increase_num_observable()
            lm.num_observable += 1
        
        # found_candidate = False

        # for lm in self.local_keyframes:
            # if lm.identifier_in_local_lm_search_ == frame.frame_id:
                # continue
        observations = self.observable_in_frame(frame)

        print("Found %d observations".format(len(observations)))
        
        # acquire more 2D-3D matches by projecting the local landmarks to the current frame
        # match::projection projection_matcher(0.8);
        # const float margin = (curr_frm_.id_ < last_reloc_frm_id_ + 2)
        #                     ? 20.0 : ((camera_->setup_type_ == camera::setup_type_t::RGBD)
        #                             ? 10.0 : 5.0);
        # projection_matcher.match_frame_and_landmarks(curr_frm_, local_landmarks_, margin);
        margin = 5
        num_matches = self.slam_matcher.match_frame_to_landmarks(frame, self.local_landmarks, margin)
        print("num_matches: ", num_matches)
        return num_matches

    def observable_in_frame(self, frame: Frame):
        """ Similar to frame.can_observe in OpenVSlam
        """
        pose_world_to_cam = frame.world_pose
        cam_center = frame.world_pose.get_origin()
        # found_candidate = False
        observations = []
        for lm in self.local_landmarks:
            if lm.identifier_in_local_lm_search_ == frame.frame_id:
                continue
            # check if observeable
            p = self.reconstruction.points[lm.lm_id]
            
            camera_point = pose_world_to_cam.transform(p)
            if camera_point[2] <= 0.0:
                continue
            point2D = self.camera.project(camera_point)
            #TODO: check boundaries?
            cam_to_lm_vec = p - cam_center
            dist = np.norm(cam_to_lm_vec)

            #TODO: Check feature scale?
            # Compute normal
            mean_normal = lm.mean_normal()
            ray_cos = np.dot(cam_to_lm_vec, mean_normal)/cam_to_lm_vec
            if ray_cos < 0.5:
                continue
            observations.append(point2D)
            # found_candidate = True

            #TODO: scale_level
            # pred_scale_lvl = lm.predict_scale_level(dist, )

            # return True, point2D
        return observations


    def mapping_with_new_keyframe(self, keyframe : Frame):
        """
        - Removes redundant frames
        - Creates new!! landmarks create_new_landmarks()
        - updates keyframe
        """
        # // set the origin keyframe -> whatever that means?
        # local_map_cleaner_->set_origin_keyframe_id(map_db_->origin_keyfrm_->id_);
        self.curr_kf = keyframe
        # // store the new keyframe to the database
        # store_new_keyframe();
        self.store_new_keyframe()

        # // remove redundant landmarks
        # local_map_cleaner_->remove_redundant_landmarks(cur_keyfrm_->id_);
        self.remove_redundant_landmarks()

        self.create_new_landmarks

    def create_new_landmarks(self, data):
        num_covisibilites = 10
        #TODO: get top n covisibilites
        curr_cam_center = self.curr_kf.pose.get_camera_center()
        covisibilites = []
        for neighbor_kfm in covisibilites:
            # neighbor_kfm = cv
            kf_cam_center = neighbor_kfm.pose.get_camera_center()
            baseline = kf_cam_center - curr_cam_center
            dist = baseline.norm()
            #if monocular
            median_depth = neighbor_kfm.compute_median_depth(True)
            if dist < 0.02 * median_depth:
                continue

            #TODO: Essential solver between two frames

            matches = self.slam_matcher.match_for_triangulation(neighbor_kfm,
                                                                self.curr_kf)
            triangulate_with_two_kfs(self.curr_kf, neighbor_kfm, matches)
        return True
    
    def triangulate_with_two_kfs(frame1: Frame, frame2: Frame, matches):

        for (m1,m2) in matches:
            #if triangulate

            lm = Landmark(self.n_landmarks)
            lm.add_observations(frame1, m1)
            lm.add_observations(frame2, m2)
            
            frame1.add_landmark(lm, m1)
            frame2.add_landmark(lm, m2)
        pass

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
        curr_lms = self.curr_kf.visible_landmarks

        for lm_id in curr_lms:
            lm = self.graph.node[lm_id]['data']
            # lm.is_observed_in_keyframe(self.curr_kf.kf_id)
            if self.curr_kf.id in lm.observations:
                map_cleaner.add_fresh_landmark
            else:
                print("lm: ", lm)
                print("lm: ", lm.observations)
                lm.observations[self.curr_kf.id] = lm_id
                lm.update_normal_and_depth()
                lm.compute_descriptor()
        
        #TODO: update graph connections
        #TODO: self.add_keyframe_to_map(self.curr_kf)
        # Is that necessary
            
    # def add_keyframe_to_map(self):
    #     self.

    def add_fresh_landmark(self, lm: Landmark):
        self.fresh_landmarks.append(lm)

    # optimize_current_frame_with_local_map
    def track_with_local_map(self, frame: Frame, slam_tracker):
        print("track_with_local_map")
        n_matches = self.search_local_landmarks()
        # create observations

        # slam_tracker.bundle_tracking(self.local_landmarks, )

        num_tracked_lms = 0
        #filter outliers and count tracked lms
        for lm in frame:
            lm.num_observable += 1
            num_tracked_lms += 1

        if num_tracked_lms < self.num_tracked_lms_thr:
            return False
        
        return True

    def new_kf_needed(self, num_tracked_lms, frame):
        """Return true if a new keyframe is needed based on the OpenVSLAM criteria
        """
        # Count the number of 3D points observed from more than 3 viewpoints
        min_obs_thr = 3 if 3 <= self.n_keyframes else 2

        # #essentially the graph
        # #find the graph connections
        # #it's about the observations in all frames and not just the kfs
        # #so we can't use the graph of only kfs
        # num_reliable_lms = get_tracked_landmarks(min_obs_thr)
        num_reliable_lms = self.curr_kf.\
            get_num_tracked_landmarks(min_obs_thr, self.graph)
        
        max_num_frms_ = 30  # the fps
        min_num_frms_ = 0

        frm_id_of_last_keyfrm_ = self.curr_kf.id
        # frame.id
        # ## mapping: Whether is processing
        # #const bool mapper_is_idle = mapper_->get_keyframe_acceptability();
        # Condition A1: Add keyframes if max_num_frames_ or more have passed
        # since the last keyframe insertion
        cond_a1 = (frm_id_of_last_keyfrm_ + max_num_frms_ <= frame.id)
        # Condition A2: Add keyframe if min_num_frames_ or more has passed
        # and mapping module is in standby state
        cond_a2 = (frm_id_of_last_keyfrm_ + min_num_frms_ <= frame.id)
        # Condition A3: Add a key frame if the viewpoint has moved from the
        # previous key frame
        cond_a3 = num_tracked_lms < (num_reliable_lms * 0.25)

        # Condition B: (Requirement for adding keyframes)
        # Add a keyframe if 3D points are observed above the threshold and
        # the percentage of 3D points is below a certain percentage
        cond_b = (self.num_tracked_lms_thr <= num_tracked_lms) and \
                 (num_tracked_lms < num_reliable_lms * self.lms_ratio_thr)

        # # Do not add if B is not satisfied
        if not cond_b:
            return False

        # # Do not add if none of A is satisfied
        if not cond_a1 and not cond_a2 and not cond_a3:
            return False

        return True
