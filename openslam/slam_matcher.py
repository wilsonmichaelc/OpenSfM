from opensfm import matching
import logging
import numpy as np
from slam_frame import Frame
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

    def match_landmarks_to_image(self, last_frame: Frame, frame: Frame, camera, data):
        # think about simply passing the descriptors of the last frame
        # for now, load
        # im1_matches = {}
        # im1_matches[curr_frame] = matching.match(ref_frame, curr_frame,
                                                #  camera, camera, data)
        cameras = data.load_camera_models()
        camera_obj = next(iter(cameras.values()))
        print("Last frame: ", last_frame)
        print("match_landmarks_to_image", last_frame.im_name, frame.im_name)
        success, matches = self.match(data, last_frame.im_name, frame.im_name, camera_obj)
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



    def matchOpenVSlam(self):
        return True
        #think about the matching.
        #reproject landmarks visible in last frame to current frame
        #under a velocity model
        #openvslam, projection.cc, l84
        #reproject to image -> check inside
        #find features in cell
        #hamming matching
