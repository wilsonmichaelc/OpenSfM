from opensfm import matching
# from feature_loader import instance
from opensfm import feature_loader
import logging
import numpy as np
from slam_types import Frame
from slam_types import Keyframe
import slam_debug
from opensfm import reconstruction
logger = logging.getLogger(__name__)

#TODO: implement an instance similar to feature loader!


class SlamMatcher(object):
    """The SlamMatcher matches a keyframe to the current frame and return the matches
    """
    def __init__(self, config):
        print("Init SlamMatcher")

    def match(self, data, ref_frame: str, curr_frame: str, camera):
        print("Matching!", ref_frame, curr_frame)
        im1_matches = matching.match(ref_frame, curr_frame,
                                     camera, camera, data)
        print("len(im1_matches) ", len(im1_matches))
        if len(im1_matches) < 30:
            return False, []
        return True, im1_matches

    # def match_frame_to_frame(self, last_frame: Frame, frame: Frame, 
    #                          camera, data):
    #     # think about simply passing the descriptors of the last frame
    #     # for now, load
    #     cameras = data.load_camera_models()
    #     camera_obj = next(iter(cameras.values()))
    #     print("Last frame: ", last_frame)
    #     print("match_frame_to_frame", last_frame.im_name, frame.im_name)
    #     success, matches = self.match(data, last_frame.im_name, frame.im_name,
    #                                   camera_obj)
    #     if success:
    #         m1, idx1, idx2 = np.intersect1d(last_frame.idx_valid, matches[:, 0],
    #                                         return_indices=True)
    #         return m1, idx1, idx2, matches

    #     return None, None, None

    # def match_kf_to_landmarks(self, frame: Keyframe, landmarks, margin, data, graph):
    #     print("match_frame_to_landmarks frame: ", frame.im_name)
    #     f2 = []
    #     for lm_id in landmarks:
    #         lm = graph.node[lm_id]['data']
    #         if lm.descriptor is not None:
    #             f2.append(lm.descriptor)
    #     f2 = np.asarray(f2)
    #     f1 = frame.descriptors
    #     print("f1, f2: ", len(f1), len(f2), f1.shape, f2.shape)
    #     matches = matching.match_brute_force_symmetric(f1, f2, data.config)
    #     if matches is None:
    #         return None
    #     return np.array(matches, dtype=int)

    def match_frame_to_landmarks(self, frame: Frame, landmarks, margin, data, graph):
        """Matches a frame to landmarks
        """
        print("match_frame_to_landmarks frame obj: ", frame)
        print("match_frame_to_landmarks frame: ", frame.im_name)
        _, f1, _ = feature_loader.instance.load_points_features_colors(
            data, frame.im_name, masked=True)
        f2 = []
        for lm_id in landmarks:
            # print("match frame to lm lm_id", lm_id)
            lm = graph.node[lm_id]['data']
            if lm.descriptor is not None:
                f2.append(lm.descriptor)
            # print("frame: ", frame.im_name, "slm_id: ", lm_id)
        f2 = np.asarray(f2)
        print("f1, f2: ", len(f1), len(f2), f1.shape, f2.shape)
        chrono = reconstruction.Chronometer()
        matches = matching.match_brute_force_symmetric(f1, f2, data.config)
        chrono.lap('frame_to_lm')
        slam_debug.avg_timings.addTimes(chrono.laps_dict)
        # for m1, m2 in matches:
        #     print("frame: ", frame.im_name, "m1: ", m1, " m2: ", m2)
        # TODO: Do some additional checks
        # indexes = [2, 3, 5]
        # for index in sorted(matches[:, 1], reverse=True):
        # del landmarks[index]

        return np.asarray(matches)  # len(matches), matches

    def match_for_triangulation(self, curr_kf: Keyframe,
                                other_kf: Keyframe, graph, data):
        # matches = []
        cameras = data.load_camera_models()
        camera_obj = next(iter(cameras.values()))
        print("match_for_triangulation", other_kf, curr_kf.im_name)
        success, matches = self.match(data, curr_kf.im_name, other_kf,
                                      camera_obj)
        return matches if success else None

    def match_for_triangulation_fast(self, curr_kf: Keyframe,
                                     other_kf: Keyframe, graph, data):
        cameras = data.load_camera_models()
        camera_obj = next(iter(cameras.values()))
        print("match_for_triangulation", other_kf.im_name, curr_kf.im_name)
        f1, f2 = curr_kf.descriptors, other_kf.descriptors
        i1, i2 = curr_kf.index, other_kf.index
        p1, p2 = curr_kf.points, other_kf.points
        config = data.config
        matches = matching.match_flann_symmetric(f1, i1, f2, i2, config)
        if matches is None:
            return None
        matches = np.asarray(matches)
        rmatches = matching.robust_match(p1, p2, camera_obj, camera_obj, 
                                         matches, config)
        rmatches = np.array([[a, b] for a, b in rmatches])
        print("n_matches {} <-> {}: {}, {}".format(
              curr_kf.im_name, other_kf,
              len(matches), len(rmatches)))
            # From indexes in filtered sets, to indexes in original sets of features
        m1 = feature_loader.instance.load_mask(data, curr_kf.im_name)
        m2 = feature_loader.instance.load_mask(data, other_kf.im_name)
        if m1 is not None and m2 is not None:
            rmatches = matching.unfilter_matches(rmatches, m1, m2)
        return np.array(rmatches, dtype=int)

    def matchOpenVSlam(self):
        return True
        #think about the matching.
        #reproject landmarks visible in last frame to current frame
        #under a velocity model
        #openvslam, projection.cc, l84
        #reproject to image -> check inside
        #find features in cell
        #hamming matching
