from opensfm import types
from opensfm import reconstruction
import networkx as nx
import logging
logger = logging.getLogger(__name__)


class SlamMapper(object):

    def __init__(self, data, config):
        """SlamMapper holds a local and global map
        """
        # self.num_keyfrms = 0
        self.reconstruction = None
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
        self.last_frame = None
        self.last_keyframe = None
        # dict because frames can be deleted
        self.keyframes = {}     # holds the id, Frame()
        self.frames = {}        # holds the id, Frame()
 

    def estimate_pose(self):
        if self.last_keyframe is not None:
            return self.last_keyframe.world_pose
        return types.Pose()

    def set_last_keyframe(self, keyframe):
        """Sets a new keyframe

        Arguments:
            keyframe: of type Frame
        """
        self.last_keyframe = keyframe
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
        print("update_local_keyframes")

    def update_local_landmarks(self):
        print("update_local_landmarks")
    
    def set_local_landmarks(self):
        print("set_local_landmarks()")
        
    def update_local_map(self):
        #Todo: unify update_local_kf, keyframes and set
        self.update_local_keyframes()
        self.update_local_landmarks()
        self.set_local_landmarks()
        return True
    
    def track_with_local_map(self):
        print("track_with_local_map")

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
        num_reliable_lms = self.last_keyframe.\
            get_num_tracked_landmarks(min_obs_thr, self.graph)
        
        max_num_frms_ = 30  # the fps
        min_num_frms_ = 0

        frm_id_of_last_keyfrm_ = self.last_keyframe.id
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
