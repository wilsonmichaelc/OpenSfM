from opensfm import reconstruction
from slam_types import Frame
from slam_types import Keyframe
from slam_types import Landmark
from opensfm import types
from opensfm import features
import logging
import cslam
logger = logging.getLogger(__name__)
import numpy as np
import networkx as nx
import slam_debug
class SlamMapper(object):

    def __init__(self, guided_matcher, data, config_slam, camera):
        """SlamMapper holds a local and global map
        """
        self.data = data
        self.camera = camera
        self.config = data.config
        self.config_slam = config_slam
        self.velocity = types.Pose()
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
        self.num_tracked_lms_thr = 15
        self.lms_ratio_thr = 0.9
        self.num_tracked_lms = 0
        self.curr_kf = None
        self.map_cleaner = cslam.LocalMapCleaner()
        self.guided_matcher = guided_matcher

    def create_init_map(self, graph_inliers, rec_init,
                        init_frame: Frame, curr_frame: Frame,
                        init_pdc=None, other_pdc=None):
        """The graph contains the KFs/shots and landmarks.
        Edges are connections between keyframes and landmarks and
        basically "observations"
        """
        # self.last_frame = curr_frame
        # Create the keyframes
        kf1 = cslam.KeyFrame(0, init_frame.cframe)
        kf2 = cslam.KeyFrame(1, curr_frame.cframe)
        
        kf1_pose = rec_init.shots[init_frame.im_name].pose.get_Rt()
        kf1_pose = np.vstack((kf1_pose, np.array([0, 0, 0, 1])))
        kf1.set_pose(np.linalg.inv(kf1_pose))
        kf2_pose = rec_init.shots[curr_frame.im_name].pose.get_Rt()
        kf2_pose = np.vstack((kf2_pose, np.array([0, 0, 0, 1])))
        kf2.set_pose(np.linalg.inv(kf2_pose))
        self.graph = graph_inliers
        self.reconstruction = rec_init
        init_frame.frame_id = 0
        # Create keyframes
        init_frame.world_pose = rec_init.shots[init_frame.im_name].pose
        self.init_frame = Keyframe(init_frame, self.data, 0)
        # self.init_frame.world_pose = \
        #     rec_init.shots[init_frame.im_name].pose
        curr_frame.frame_id = 1
        curr_frame.world_pose = rec_init.shots[curr_frame.im_name].pose
        curr_kf = Keyframe(curr_frame, self.data, 1)
        # curr_kf.world_pose = rec_init.shots[curr_frame.im_name].pose

        self.init_frame.ckf = kf1
        curr_kf.ckf = kf2
         # Add to data and covisibility
        self.add_keyframe(self.init_frame)
        self.add_keyframe(curr_kf)
        # self.n_frames = 2
        max_lm = 0  # find the highest lm id
        n_lm_added = 0
        c_lms = []
        self.update_with_last_frame(init_frame)
        self.update_with_last_frame(curr_frame)
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
        self.num_tracked_lms = len(self.local_landmarks)
        # TODO: Scale map such that the median of depths is 1.0!
        curr_frame.world_pose = curr_kf.world_pose

    def update_with_last_frame(self, frame: Frame):
        """Updates the last frame and the related variables in slam mapper
        """
        if self.n_frames > 0: # we alread have frames
            self.velocity = frame.world_pose.compose(self.last_frame.world_pose.inverse())
            self.pre_last = self.last_frame
        self.n_frames += 1
        self.last_frame = frame
        # # debug
        # p0_3D = np.zeros([len(self.graph[self.init_frame.im_name]), 3], dtype=np.float32)
        # p0 = np.zeros([len(self.graph[self.init_frame.im_name]), 2], dtype=np.float32)
        # for idx, lm_id in enumerate(self.graph[self.init_frame.im_name]):
        #     p0_3D[idx, :] = self.reconstruction.points[str(lm_id)].coordinates
        #     p0[idx, :] = self.graph.get_edge_data(self.init_frame.im_name, str(lm_id))['feature']
        # im1, im2 = self.data.load_image(self.init_frame.im_name), self.data.load_image(frame.im_name)
        # im3 = self.data.load_image(self.last_frame.im_name)
        # # project landmarks into kf1
        # cam  = self.camera[1]
        # slam_debug.disable_debug = False
        # # camera_point = self.init_frame.world_pose.transform_many(p0_3D)
        # camera_point = frame.world_pose.transform_many(p0_3D)
        # p1 = cam.project_many(camera_point)
        # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        # slam_debug.visualize_matches_pts(p0, p1, np.column_stack((a, a)), im1, im2, False, title="to frame"+frame.im_name)
        # # project landmarks into kf2
        # camera_point2 =self.last_frame.world_pose.transform_many(p0_3D)
        # p12 = cam.project_many(camera_point2)
        # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        # slam_debug.visualize_matches_pts(p0, p12, np.column_stack((a, a)), im1, im3, False, title="to last frame"+self.last_frame.im_name)
        # # project landmarks into coordinate system of kf 1 and then to kf2
        # camera_point3 = self.velocity.compose(self.last_frame.world_pose).transform_many(p0_3D)
        # p13 = cam.project_many(camera_point3)
        # a = np.asarray(np.arange(0,len(p0)), dtype=int)
        # slam_debug.visualize_matches_pts(p0, p13, np.column_stack((a, a)), im1, im2, True, title="to last frame and then frame")
        # # debug end
        # slam_debug.disable_debug = True
        
        # self.num_tracked_lms = len(self.last_lk)
        # print("self.num_tracked_lms {} vs lms in last kf {}, ratio {}".format(self.num_tracked_lms, len(self.graph[self.keyframes[-1]]), self.num_tracked_lms/len(self.graph[self.keyframes[-1]])))
        # now toogle all the landmarks as observed
        # for lm_id, _ in frame.lk_landmarks_:
        #     # print("Trying to fetch: ", lm_id)
        #     lm = self.graph.node[lm_id]['data']
        #     lm.num_observable += 1
        # print("Update with last frame {}, {}".format(self.n_frames, len(frame.landmarks_)))
    

    def new_keyframe_is_needed(self, frame: Frame):
        num_keyfrms = len(self.local_keyframes)
        # const unsigned int min_obs_thr = (3 <= num_keyfrms) ? 3 : 2;
        # const auto num_reliable_lms = ref_keyfrm.get_num_tracked_landmarks(min_obs_thr);
        min_obs_thr = 3 if (3 <= num_keyfrms) else 2
        last_kf = self.local_keyframes[-1]
        num_reliable_lms = last_kf.get_num_tracked_lms(min_obs_thr)
        max_num_frms_ = 10  # the fps
        min_num_frms_ = 2
        # if frame.frame_id > 15 and frame.frame_id % 3:
        #     return True
        frm_id_of_last_keyfrm_ = self.curr_kf.kf_id
        print("curr_kf: ", self.curr_kf.kf_id, self.curr_kf.frame_id)
        print("frame.frame_id: ", frame.frame_id, frm_id_of_last_keyfrm_)
        # frame.id
        # ## mapping: Whether is processing
        # #const bool mapper_is_idle = mapper_->get_keyframe_acceptability();
        # Condition A1: Add keyframes if max_num_frames_ or more have passed

        # since the last keyframe insertion
        cond_a1 = (frm_id_of_last_keyfrm_ + max_num_frms_ <= frame.frame_id)
        # Condition A2: Add keyframe if min_num_frames_ or more has passed
        # and mapping module is in standby state
        cond_a2 = (frm_id_of_last_keyfrm_ + min_num_frms_ <= frame.frame_id)
        # cond_a2 = False
        # Condition A3: Add a key frame if the viewpoint has moved from the
        # previous key frame
        cond_a3 = self.num_tracked_lms < (num_reliable_lms * 0.25)

        print("self.num_tracked_lms_thr {} self.num_tracked_lms {}\n \
               num_reliable_lms {} * self.lms_ratio_th={}".
               format(self.num_tracked_lms_thr, self.num_tracked_lms,
                      num_reliable_lms, num_reliable_lms * self.lms_ratio_thr))
        # Condition B: (Requirement for adding keyframes)
        # Add a keyframe if 3D points are observed above the threshold and
        # the percentage of 3D points is below a certain percentage
        cond_b = (self.num_tracked_lms_thr <= self.num_tracked_lms) and \
                 (self.num_tracked_lms < num_reliable_lms * self.lms_ratio_thr)

        print("cond_a1: {}, cond_a2: {}, cond_a3: {}, cond_b: {}"
                .format(cond_a1, cond_a2, cond_a3, cond_b))
        # # Do not add if B is not satisfied
        if not cond_b:
            print("not cond_b -> no kf")
            return False
        
        # # Do not add if none of A is satisfied
        if not cond_a1 and not cond_a2 and not cond_a3:
            print("not cond_a1 and not cond_a2 and not cond_a3 -> no kf")
            return False
        print("NEW KF", frame.im_name)
        # exit()
        return True

    def insert_new_keyframe(self, frame: Frame):


        # Create new Keyframe
        new_kf = Keyframe(frame, self.data, self.n_keyframes)
        new_kf.ckf = cslam.KeyFrame(new_kf.kf_id, frame.cframe)
        kf1_pose = np.vstack((new_kf.world_pose.get_Rt(), np.array([0, 0, 0, 1])))
        # kf1.set_pose(np.linalg.inv(kf1_pose))
        new_kf.ckf.set_pose(np.linalg.inv(kf1_pose))
        self.add_keyframe(new_kf)
        
        # TODO: Add observations to graph!
        valid_ids = frame.cframe.get_valid_idx()
        valid_kps = frame.cframe.get_valid_keypts()
        valid_lms = frame.cframe.get_valid_lms()
        colors = frame.colors
        valid_kps_norm, _, _ = features.normalize_features(valid_kps, None, None, self.camera[1].width, self.camera[1].height)
        graph = self.graph
        for f_id, kp, lm in zip(valid_ids, valid_kps_norm, valid_lms):
            x, y, s = kp
            r, g, b = colors[f_id, :]
            # add the edge
            graph.add_edge(new_kf.im_name, str(lm.lm_id),
                       feature=(float(x), float(y)),
                       feature_scale=float(s),
                       feature_id=int(f_id),
                       feature_color=(float(r), float(g), float(b)))

        #Now, count 
        print("len(graph[new_kf.im_name]): ", len(graph[new_kf.im_name]),
              "len(valid_lms): ", len(valid_lms))
        
        # // set the origin keyframe
        # local_map_cleaner_->set_origin_keyframe_id(map_db_->origin_keyfrm_->id_);


        if self.n_keyframes % 15 == 0:
            self.paint_reconstruction()
            self.save_reconstruction(frame.im_name+"aft")

        # lms_clean = new_kf.ckf.update_lms_after_kf_insert() # == store_new_keyframe
        self.map_cleaner.update_lms_after_kf_insert(new_kf.ckf)
        self.map_cleaner.remove_redundant_lms(new_kf.kf_id)
        self.create_new_landmarks()
    
    def create_new_landmarks(self):
        """Creates a new landmarks with using the newly added KF
        """
        # Again, just take the last 10 frames
        # but not the current one!
        local_keyframes = self.local_keyframes[-10:-1]
        new_kf = self.local_keyframes[-1]
        new_im = self.data.load_image(self.keyframes[-1])
        local_im_names = self.keyframes[-10:-1]
        new_cam_center = new_kf.get_cam_center()
        new_Tcw = np.linalg.inv(new_kf.get_pose())
        new_R = new_Tcw[0:3, 0:3]
        new_t = new_Tcw[0:3, 3]
        print("local_kf: ", len(local_keyframes))
        # TODO! check new_kf pose
        for (old_kf, im_name) in zip(local_keyframes, local_im_names):
            old_cam_center = old_kf.get_cam_center()
            baseline_vec = old_cam_center - new_cam_center
            baseline_dist = np.linalg.norm(baseline_vec)
            median_depth_in_old = old_kf.compute_median_depth(True)
            if baseline_dist < 0.02 * median_depth_in_old:
                continue

            # Compute essential matrix!
            old_Tcw = np.linalg.inv(old_kf.get_pose())
            old_R = old_Tcw[0:3, 0:3]
            old_t = old_Tcw[0:3, 3]

            E_old_to_new = self.guided_matcher.create_E_21(new_R, new_t, old_R, old_t)
            matches = self.guided_matcher.match_for_triangulation(new_kf, old_kf, E_old_to_new)
            # print(matches)
            old_im = self.data.load_image(im_name)
            slam_debug.disable_debug = False
            slam_debug.visualize_matches_pts(new_kf.getKptsPy(), old_kf.getKptsPy(), np.array(matches),new_im, old_im, do_show=True, is_normalized=False, title=im_name)
            slam_debug.disable_debug = False
        
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
            # add kf object to existing graph node
            self.graph.add_node(str(kf.im_name), bipartite=0, data=kf)

        self.keyframes.append(kf.im_name)
        self.local_keyframes.append(kf.ckf)
        self.curr_kf = kf

    def add_landmark(self, lm: Landmark):
        """Add landmark to graph"""

    def paint_reconstruction(self):
        if self.reconstruction is not None and self.graph is not None:
            reconstruction.paint_reconstruction(self.data, self.graph,
                                                self.reconstruction)

    def save_reconstruction(self, name: str):
        if self.reconstruction is not None:
            logger.debug("Saving reconstruction with {} points and {} frames".
                         format(len(self.reconstruction.points),
                                len(self.reconstruction.shots)))
            self.data.save_reconstruction([self.reconstruction],
                                     'reconstruction' + name + '.json')
