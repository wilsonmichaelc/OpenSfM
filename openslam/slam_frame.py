from opensfm import feature_loader
from opensfm import features
from opensfm import types
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Frame(object):

    def __init__(self, name):
        print("Creating frame: ", name)
        self.im_name = name
        self.visible_landmarks = None
        self.idx_valid = None
        self.id = -1
        self.kf_id = -1  # if non-KF, id of "parent" KF
        self.is_keyframe = False
        self.world_pose = types.Pose()
        self.rel_pose_to_kf = types.Pose()
        self.descriptors = None

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

    def create_new_landmarks(self, data):
        return True

    # def load_points_features_clr(self, data):
    #     image = self.im_name
    #     has_features = data.features_exist(image)
    #     if has_features:
    #         return feature_loader.instance.load_points_features_colors(data, image)
    #     p_unmasked, f_unmasked, c_unmasked = features.extract_features(
    #         data.load_image(image), data.config)

    #     fmask = data.load_features_mask(image, p_unmasked)

    #     p_unsorted = p_unmasked[fmask]
    #     f_unsorted = f_unmasked[fmask]
    #     c_unsorted = c_unmasked[fmask]

    #     if len(p_unsorted) == 0:
    #         logger.warning('No features found in image {}'.format(image))
    #         return

    #     size = p_unsorted[:, 2]
    #     order = np.argsort(size)
    #     p_sorted = p_unsorted[order, :]
    #     f_sorted = f_unsorted[order, :]
    #     c_sorted = c_unsorted[order, :]
    #     data.save_features(image, p_sorted, f_sorted, c_sorted)
    #     return p_sorted, f_sorted, c_sorted
