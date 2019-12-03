# from opensfm import feature_loader
# from opensfm import features
from opensfm import types
import logging
import numpy as np
# from enum import Enum
logger = logging.getLogger(__name__)


# class LandmarkStatus(Enum):
#     Valid = 1
#     Invalid = 2
#     NotClear = 3

class Landmark(object):

    def __init__(self, graph_id):
        """Creates a landmark

        Arguments:
            graph_id : The id the landmark has in the graph (must be unique)
        """
        self.lm_id = graph_id
        self.is_observable_in_tracking = False
        self.local_map_update_identifier = -1  # the last frame where it was observed
        self.identifier_in_local_lm_search_ = -1
        self.n_observable = 0  # the number of frames and KFs it is seen in
        self.descriptor = None
        self.pos_w = []  # 3D position as reconstructed
        self.num_observable = 0
        self.num_observed = 0
        self.observations = {}  # keyframes, where the landmark is observed
        self.first_kf_id = -1
        self.ref_kf = None
        # self.num_observations_
    
    def update_normal_and_depth(self, graph):
        if len(self.observations) == 0:
            return
        self.mean_normal = []
        for kf, idx in self.observations.values():
            normal = self.pos_w - kf.pose.world_pose()
            mean_normal += normal / normal.norm()

        # n_observations = len(self.observations)
        # cam_to_lm_vec = self.pos_w - self.ref_kf.pose.world_pose()
        # dist = cam_to_lm_vec.norm()
        #TODO: scale level, scale factor, depth
        self.mean_normal /= len(self.observations)

    def prepare_for_erasing(self):
        pass

    def add_observation(self, kf, idx):
        """ Adds an observation

        An observation is defined by the kf and the id of the feature in this kf
        """
        if kf in self.observations:
            return
        self.observations[kf] = idx
        self.num_observations += 1

    def compute_descriptor(self, graph):
        """ Computes the descriptor from the observations
        - similar to OpenVSlam
        - or simply take the most recent one
        """
        """Computes the descriptor of the lm
        from all the observations
        Take random descriptor
        """

        keyframes = graph[str(self.lm_id)]
        # descriptors = []
        for kf_name in keyframes:
            kf = graph.node[kf_name]['data']
            track = graph.get_edge_data(kf_name, str(self.lm_id))
            self.descriptor = kf.descriptors[track['feature_id']]
            return
    #     if len(graph[self.lm_id]) == 0:
    #         self.descriptor = None
    #         return

    #     #Compute the descriptor
    #     for kf in graph[self.lm_id]:
    #         print("kf: ", kf)
    # # def add_observation(self, graph, keyframe):

class Frame(object):

    def __init__(self, name):
        print("Creating frame: ", name)
        self.im_name = name
        self.landmarks_ = []
        self.idx_valid = None
        self.id = -1
        self.kf_id = -1  # if non-KF, id of "parent" KF
        self.is_keyframe = False
        self.world_pose = types.Pose()
        self.rel_pose_to_kf = types.Pose()
        self.descriptors = None
        
        #stores where the frame was last updated
        self.local_map_update_identifier = -1
        
    def update_visible_landmarks(self, idx):
        if self.visible_landmarks is None:
            return
        self.visible_landmarks = self.visible_landmarks[idx, :]

    def set_visible_landmarks(self, points, inliers):
        self.visible_landmarks = points  # id, coordinates
        self.idx_valid = np.zeros(len(inliers.values()))
        for (idx, feature) in enumerate(inliers.values()):
            self.idx_valid[idx] = feature['feature_id']

    def store(self):
        """Reduces the object to just the header"""
        self.visible_landmarks = []

    def make_keyframe(self, data, world_pose, rel_pose, kf_id):
        self.kf_id = kf_id
        self.rel_pose_to_kf = rel_pose
        self.world_pose = world_pose
        self.is_keyframe = True
        _, self.descriptors, _ = data.load_features(self.im_name)

    def get_num_tracked_landmarks(self, min_obs_thr, graph):
        """Counts the number of reliable landmarks, i.e. all visible in
        greater or equal `min_obs_thr` keyframes
        """
        if min_obs_thr > 0:
            n_lms = 0
            for lm in self.visible_landmarks:
                print("lm: ", lm)
                print("graph[lm]: ", graph[lm])
                print("len graph[lm]: ", len(graph[lm]))
                if len(graph[lm]) >= min_obs_thr:
                    n_lms += 1
            return n_lms
        return len(self.visible_landmarks)

    #in case it is removed
    # def make_non_keyframe(self, )

    # def create_new_landmarks(self, data):
    #     return True


class Keyframe(object):
    def __init__(self, frame: Frame, data):
        self.landmarks_ = frame.landmarks_
        self.im_name = frame.im_name  # im_name should also be unique
        self.kf_id = -1  # unique_id
        self.landmarks = []
        self.keypts = []
        _, self.descriptors, _ = data.load_features(self.im_name)
        self.world_pose = types.Pose()
        

    def add_landmark(self, lm: Landmark, idx):
        self.landmarks_[idx] = lm
