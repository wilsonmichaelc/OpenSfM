
class SlamTracker(object):

    def __init__(self, slam_matcher):

        self.slam_matcher = slam_matcher


        print("init slam tracker")
    def track(graph, reconstruction, landmarks, last_frame, frame, camera, init_pose, config):
        """Align the current frame to the already estimated landmarks
            (visible in the last frame)
            landmarks visible in last frame
        """
        m1, idx1, idx2 = self.slam_matcher.match_landmarks_to_image(self, landmarks, frame, last_frame, camera, data)

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
