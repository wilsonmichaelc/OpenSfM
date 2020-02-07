from opensfm import reconstruction
from slam_types import Frame
from slam_types import Keyframe
from slam_types import Landmark
from opensfm import types
import logging
import cslam
logger = logging.getLogger(__name__)
import numpy as np
import networkx as nx

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
        self.last_frame  = None
        self.n_frames = 0
        self.n_keyframes = 0
        self.covisibility = nx.Graph()
        self.covisibility_list = []
        self.keyframes = []
        self.local_keyframes = []
        self.local_landmarks = []

    def create_init_map(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame,
                        init_pdc=None, other_pdc=None):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
        """
        self.last_frame = curr_frame
        # Create the keyframes
        kf1 = cslam.KeyFrame(0, init_frame.cframe)
        kf2 = cslam.KeyFrame(1, curr_frame.cframe)
        kf1_pose = rec_init.shots[init_frame.im_name].pose.get_Rt()
        kf1_pose = np.vstack((kf1_pose, np.array([0, 0, 0, 1])))
        kf1.set_pose(kf1_pose)
        kf2_pose = rec_init.shots[curr_frame.im_name].pose.get_Rt()
        kf2_pose = np.vstack((kf2_pose, np.array([0, 0, 0, 1])))
        kf2.set_pose(kf2_pose)
        self.graph = graph_inliers
        self.reconstruction = rec_init
        init_frame.frame_id = 0
        # Create keyframes
        self.init_frame = Keyframe(init_frame, self.data, 0)
        self.init_frame.world_pose = \
            rec_init.shots[init_frame.im_name].pose
        curr_frame.frame_id = 1
        curr_kf = Keyframe(curr_frame, self.data, 1)
        curr_kf.world_pose = rec_init.shots[curr_frame.im_name].pose
        self.init_frame.ckf = kf1
        curr_kf.ckf = kf2
         # Add to data and covisibility
        self.add_keyframe(self.init_frame)
        self.add_keyframe(curr_kf)
        self.n_frames = 2
        max_lm = 0  # find the highest lm id
        n_lm_added = 0
        c_lms = []
        # for i in range(0, 500):
        #     pos_w = np.array([0,0,0], dtype=np.float)
        #     clm = cslam.Landmark(int('0'), kf1, pos_w)
        #     print("clm for", clm)
        # exit()
        for lm_id in self.graph[self.init_frame.im_name]:
            lm = Landmark(int(lm_id))
            print("new lm", n_lm_added)
            pos_w = rec_init.points[str(lm_id)].coordinates
            print("new pos_w", pos_w)
            clm = cslam.Landmark(int(lm_id), kf1, pos_w)
            print("clm", clm)
            # c_lms.append(clm)
            print("new clm", n_lm_added)

            lm.clm = clm
            lm.first_kf_id = self.init_frame.kf_id
            self.graph.add_node(lm_id, data=lm)
            int_id = int(lm_id)
            if int_id > max_lm:
                max_lm = int_id
            print("bef edge", n_lm_added)
            f1_id = self.graph.get_edge_data(lm_id, self.init_frame.im_name)["feature_id"]
            f2_id = self.graph.get_edge_data(lm_id, curr_kf.im_name)["feature_id"]
            print("aft edge", n_lm_added)
            # connect landmark -> kf
            clm.add_observation(kf1, f1_id)
            print("kf1")
            clm.add_observation(kf2, f2_id)
            print("kf2")
            # connect kf -> landmark in the graph
            kf1.add_landmark(clm, f1_id)
            print("add lm kf1")
            kf2.add_landmark(clm, f2_id)
            print("add lm kf2")
            clm.compute_descriptor()
            print("cmp desc")
            clm.update_normal_and_depth()
            print("add update_normal_and_depth kf1")
            self.local_landmarks.append(lm_id)
            print("loc lm", n_lm_added)
            n_lm_added += 1
            curr_frame.cframe.add_landmark(clm, f2_id)
            print("add lm currfrm", n_lm_added)

        # for lm_id in self.graph[self.init_frame.im_name]:
        #     print("lm_id: ", lm_id)
        #     e = self.graph.get_edge_data(lm_id, self.init_frame.im_name)
        #     print("e", e)
        #     print(self.graph.node[lm_id]['data'])


        print("create_init_map: len(local_landmarks): ",
              len(self.local_landmarks), n_lm_added)
        self.current_lm_id = max_lm + 1

        # TODO: Scale map such that the median of depths is 1.0!
        curr_frame.world_pose = curr_kf.world_pose

        
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
        self.init_frame = Keyframe(init_frame, self.data, 0)
        self.init_frame.world_pose = \
            rec_init.shots[init_frame.im_name].pose
        curr_frame.frame_id = 1
        curr_kf = Keyframe(curr_frame, self.data, 1)
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
        logger.debug("Adding new keyframe ".format(kf.kf_id))
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
        self.local_keyframes.append(kf.ckf)

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
