from opensfm import reconstruction
from slam_types import Frame
from slam_types import Keyframe
from slam_types import Landmark
from opensfm import types
import logging
import cslam_types as ctypes
logger = logging.getLogger(__name__)
import numpy as np

class SlamMapper(object):

    def __init__(self, data, config_slam, camera):
        """SlamMapper holds a local and global map
        """
        self.data = data
        self.camera = camera
        self.config = data.config
        self.config_slam = config_slam

        self.reconstruction = None
        self.graph = None

        self.n_frames = 0
        self.n_keyframes = 0

    def create_init_map(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame,
                        init_pdc=None, other_pdc=None):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
        """
        # Create the keyframes
        kf1 = ctypes.KeyFrame(0, init_frame.cframe)
        kf2 = ctypes.KeyFrame(1, curr_frame.cframe)
        kf1_pose = rec_init.shots[init_frame.im_name].pose.get_Rt()
        kf1_pose = np.vstack((kf1_pose,np.array([0, 0, 0, 1])))
        kf1.set_pose(kf1_pose)
        kf2_pose = rec_init.shots[frame.im_name].pose.get_Rt()
        kf2_pose = np.vstack((kf2_pose,np.array([0, 0, 0, 1])))
        kf2.set_pose(kf2_pose)
    def create_init_map2(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame,
                        init_pdc=None, other_pdc=None):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
        """

        lm_id = 0
        lm_c = ctypes.Landmark(lm_id)
        kf1 = ctypes.KeyFrame(0, init_frame.cframe)
        kf2 = ctypes.KeyFrame(1, curr_frame.cframe)

        self.graph = graph_inliers
        self.reconstruction = rec_init
        init_frame.frame_id = 0
        # Create keyframes
        self.init_frame = Keyframe(init_frame)
        self.init_frame.world_pose = \
            rec_init.shots[init_frame.im_name].pose
        curr_frame.frame_id = 1
        curr_kf = Keyframe(curr_frame)
        curr_kf.world_pose = rec_init.shots[curr_frame.im_name].pose

         # Add to data and covisibility
        self.add_keyframe(self.init_frame)
        self.add_keyframe(curr_kf)
        self.n_frames = 2
        max_lm = 0  # find the highest lm id
        n_lm_added = 0

        # Add landmark objects to nodes
        for lm_id in self.graph[self.init_frame.im_name]:
            lm = Landmark(int(lm_id))
            lm.num_observable = 2
            lm.num_observed = 2
            lm.first_kf_id = self.init_frame.kf_id
            self.graph.add_node(lm_id, data=lm)

            int_id = int(lm_id)
            if int_id > max_lm:
                max_lm = int_id
            lm.compute_descriptor(curr_kf, self.graph) # update descriptor
            pos_w = rec_init.points[str(lm_id)].coordinates # update normal
            lm.update_normal_and_depth(pos_w, self.graph)
            self.local_landmarks.append(lm_id)
            n_lm_added += 1

        logger.debug("create_init_map: len(local_landmarks): %, %".
                     format(len(self.local_landmarks), n_lm_added))
        self.current_lm_id = max_lm + 1
        # TODO: think about the local landmarks

    def add_keyframe(self, kf: Keyframe):
        """Adds a keyframe to the map graph
        and the covisibility graph
        """
        logger.debug("Adding new keyframe ", kf.kf_id)
        # add kf object to existing graph node
        self.graph.add_node(str(kf.im_name), bipartite=0, data=kf)
        # and to covisibilty graph
        self.covisibility.add_node(str(kf.im_name))
        self.covisibility_list.append(str(kf.im_name))
        self.n_keyframes += 1
        
        if not self.graph.has_node(kf.im_name):
            # create new shot
            shot1 = types.Shot()
            shot1.id = kf.im_name
            shot1.camera = self.camera[1]
            shot1.pose = kf.world_pose
            shot1.metadata = reconstruction.\
                get_image_metadata(self.data, kf.im_name)
            self.reconstruction.add_shot(shot1)

        self.keyframes.append(kf.im_name)
        self.local_keyframes.append(kf.im_name)

    def add_landmark(self, lm: Landmark):
        """Add landmark to graph"""

    def paint_reconstruction(self, data):
        if self.reconstruction is not None and self.graph is not None:
            reconstruction.paint_reconstruction(self.data, self.graph,
                                                self.reconstruction)

    def save_reconstruction(self, data, name: str):
        if self.reconstruction is not None:
            logger.debug("Saving reconstruction with {} points and {} frames".
                         format(len(self.reconstruction.points),
                                len(self.reconstruction.shots)))
            data.save_reconstruction([self.reconstruction],
                                     'reconstruction' + name + '.json')
