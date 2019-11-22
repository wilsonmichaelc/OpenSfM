import networkx as nx
import logging
logger = logging.getLogger(__name__)
class SlamMapper(object):

    def __init__(self, data, config):
        """SlamMapper holds a local and global map
        """
        self.num_keyfrms = 0
        
        # Threshold of the ratio of the number of 3D points observed in the current frame to the number of 3D points observed in the latest key frame
        self.num_tracked_lms_thr = 15
        self.lms_ratio_thr = 0.9
        self.n_tracks = 0
        self.graph = nx.Graph()

    def add_new_tracks(self, im1, im2, matches):
        for m in matches:
            self.graph.add_node(str(im1), bipartite=0)
            self.graph.add_node(str(im2), bipartite=0)
            #figure out the feature id
            fid1 = 0
            fid2 = 1
            self.graph.add_node(str(fid1), bipartite=1)
            self.graph.add_node(str(fid2), bipartite=1)
            self.graph.add_edge(str(image),
                                str(track_id),
                                feature=(float(x), float(y)),
                                feature_scale=float(s),
                                feature_id=int(featureid),
                                feature_color=(float(r), float(g), float(b)))
            # Add the edges
        return True

    def clean_landmarks(self):
        return True

    def new_kf_needed(self, num_tracked_lms):
        """Return true if a new keyframe is needed based on the OpenVSLAM criteria
        """
        return True
        # # reference keyframe: Count the number of 3D points observed from more than 3 viewpoints
        # min_obs_thr = 3 if 3 <= num_keyfrms else 2
        # #essentially the graph
        # #find the graph connections
        # #it's about the observations in all frames and not just the kfs
        # #so we can't use the graph of only kfs
        # num_reliable_lms = get_tracked_landmarks(min_obs_thr)

        
        # ## mapping: Whether is processing
        # #const bool mapper_is_idle = mapper_->get_keyframe_acceptability();



        # # Condition A1: Add keyframes if max_num_frames_ or more have passed since the last keyframe insertion
        # cond_a1 = (frm_id_of_last_keyfrm_ + max_num_frms_ <= curr_frm.id_)
        # # Condition A2: Add keyframe if min_num_frames_ or more has passed and mapping module is in standby state
        # cond_a2 = (frm_id_of_last_keyfrm_ + min_num_frms_ <= curr_frm.id_) #&& mapper_is_idle
        # # Condition A3: Add a key frame if the viewpoint has moved from the previous key frame
        # cond_a3 = num_tracked_lms < (num_reliable_lms * 0.25)

        # # Condition B: (Requirement for adding keyframes) Add a keyframe if 3D points are observed above the threshold and the percentage of 3D points is below a certain percentage
        # cond_b = (num_tracked_lms_thr <= num_tracked_lms) and (num_tracked_lms < num_reliable_lms * lms_ratio_thr);

        # # Do not add if B is not satisfied
        # if not cond_b: 
        #     return False

        # # Do not add if none of A is satisfied
        # if not cond_a1 and not cond_a2 and not cond_a3:
        #     return False

        # if (mapper_is_idle) {
        #     # mapping module: If is not in process, add keyframe for now
        #     return True;
        # }

        # mapping module: If is processing, stop local BA and add keyframe
        # if (setup_type_ != camera::setup_type_t::Monocular
        #     && mapper_->get_num_queued_keyframes() <= 2) {
        #     mapper_->abort_local_BA();
        #     return true;
        # }

        return False
