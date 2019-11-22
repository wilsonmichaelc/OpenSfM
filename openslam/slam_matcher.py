from opensfm import matching
import logging
logger = logging.getLogger(__name__)

class SlamMatcher(object):
    """The SlamMatcher matches a keyframe to the current frame and return the matches
    """
    def __init__(self, config):
        print("Init SlamMatcher")

    def match(self, data, ref_frame, curr_frame, camera):
        print("Matching!")
        # matches = matching.match_images(data, [ref_frame], [curr_frame])
        # print(matches)
        im1_matches = {}

    
        im1_matches[curr_frame] = matching.match(ref_frame, curr_frame,
                                                 camera, camera, data)
        # print(im1_matches)
        # print(len(matches))
        print(len(im1_matches[curr_frame]))
        # print(len(matches[0][(ref_frame, curr_frame)]))
        # for im1, im2 in matches:
            # print ("im1: ",im1, " im2: ", im2, " m: ", matches[im1, im2])
        # print(matches.values(), len(matches.values()))
        
        
        # len(im1_matches[1]))
        num_matches = sum(1 for m in im1_matches.values() if len(m) > 0)
        print(num_matches)
        logger.info('Image {} matches: {} out of 2'.
                     format(ref_frame, num_matches)) #, len(candidates)))
        if len(im1_matches[curr_frame]) < 100:
            return False, {}
        return True, im1_matches

    def matchOpenVSlam(self):
        return True
        #think about the matching.
        #reproject landmarks visible in last frame to current frame
        #under a velocity model
        #openvslam, projection.cc, l84
        #reproject to image -> check inside
        #find features in cell
        #hamming matching
