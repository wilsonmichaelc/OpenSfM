from opensfm import matching
# from feature_loader import instance
from opensfm import feature_loader
import logging
import numpy as np
from slam_types import Frame
logger = logging.getLogger(__name__)

#TODO: implement an instance similar to feature loader!

class SlamMatcher(object):
    """The SlamMatcher matches a keyframe to the current frame and return the matches
    """
    def __init__(self, config):
        print("Init SlamMatcher")

    def match(self, data, ref_frame: str, curr_frame: str, camera):
        print("Matching!", ref_frame, curr_frame)
        # print("Matching!", ref_frame.im_name, curr_frame.im_name)
        im1_matches = {}
        im1_matches[curr_frame] = matching.match(ref_frame, curr_frame,
                                                 camera, camera, data)
        print("len(im1_matches[curr_frame]) ", len(im1_matches[curr_frame]))
        num_matches = sum(1 for m in im1_matches.values() if len(m) > 0)
        print("num_matches ", num_matches)
        logger.info('Image {} matches: {} out of 2'.
                    format(ref_frame, num_matches))
        if len(im1_matches[curr_frame]) < 100:
            return False, {}
        return True, im1_matches

    def match_current_and_last_frame(self, last_frame: Frame, frame: Frame,
                                     camera, data):
        cameras = data.load_camera_models()
        camera_obj = next(iter(cameras.values()))
        print("Last frame: ", last_frame)
        print("Frame: ", frame)
        print("match_current_and_last_frame", last_frame.im_name, frame.im_name)
        margin = 10
        return self.match_frame_to_landmarks(frame, last_frame.landmarks_,
                                             margin, data)

    def match_frame_to_frame(self, last_frame: Frame, frame: Frame, 
                             camera, data):
        # think about simply passing the descriptors of the last frame
        # for now, load
        # im1_matches = {}
        # im1_matches[curr_frame] = matching.match(ref_frame, curr_frame,
                                                #  camera, camera, data)
        cameras = data.load_camera_models()
        camera_obj = next(iter(cameras.values()))
        print("Last frame: ", last_frame)
        print("match_frame_to_frame", last_frame.im_name, frame.im_name)
        success, matches = self.match(data, last_frame.im_name, frame.im_name,
                                      camera_obj)
        if success:
            # valid_idx == idx of lm in features
            # matches[:,0] == valid idx of matches
            #find the intersection
            matches = matches[frame.im_name]
            # print("idx_valid: ", last_frame.idx_valid, len(last_frame.idx_valid))
            # print("matches: ", matches)
            # print("matches.shape: ", matches.shape)
            # matches[last_frame.idx_valid,:]
            #return the match indices
            m1, idx1, idx2 = np.intersect1d(last_frame.idx_valid, matches[:, 0],
                                          return_indices=True)
            # m1 contains the valid matches of the features in last frame
            # idx1 contains the indices of the landmarks
            # idx2 contains the indices of the features in new frame
            # print("len(m1): ", len(m1), len(idx1), len(idx2))
            return m1, idx1, idx2, matches
        
        return None, None, None

    # def distribute_keypoints_on_pyr(self, , width, height):


    def match_frame_to_landmarks(self, frame: Frame, landmarks, margin, data):
        """Matches a frame to landmarks
        """
        print("frame obj: ", frame)
        print("frame: ", frame.im_name)
        p1, f1, _ = feature_loader.instance.load_points_features_colors(
            data, frame.im_name, masked=True)
        f2 = []
        # print(landmarks)
        for lm in landmarks:
            # print("lm: ", lm, lm.lm_id, lm.is_observable_in_tracking)
            # if lm.is_observable_in_tracking and \
            #    lm.descriptors is not None:
            if lm.descriptor is not None:
                f2.append(lm.descriptor)
        # print("f1: ", f1)
        # print("f2: ", list(f2))
        print("f1, f2: ", len(f1), len(f2))
        print("f1 type: ", type(f1))
        print("f2 type: ", type(np.asarray(f2)), type(f2))
        f2 = np.asarray(f2)
        # print("f1.dtype: ", f1.dtype)
        # print("f2.dtype: ", f2.dtype)
        matches = matching.match_brute_force_symmetric(f1, f2, data.config)
        # print(len(matches))
        # TODO: Do some additional checks
        # indexes = [2, 3, 5]
        # for index in sorted(matches[:, 1], reverse=True):
            # del landmarks[index]

        
        return matches #len(matches), matches

    def matchOpenVSlam(self):
        return True
        #think about the matching.
        #reproject landmarks visible in last frame to current frame
        #under a velocity model
        #openvslam, projection.cc, l84
        #reproject to image -> check inside
        #find features in cell
        #hamming matching
