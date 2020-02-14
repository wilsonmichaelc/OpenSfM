from opensfm import reconstruction
from slam_types import Frame
from slam_types import Keyframe
from slam_types import Landmark
from opensfm import types
from opensfm import features
from opensfm import csfm
import logging
import cslam
logger = logging.getLogger(__name__)
import numpy as np
import networkx as nx
import slam_debug
from collections import defaultdict
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
        self.n_landmarks = 0
        self.covisibility = nx.Graph()
        self.covisibility_list = []
        self.keyframes = []
        self.c_keyframes = []
        self.local_landmarks = []
        self.num_tracked_lms_thr = 15
        self.lms_ratio_thr = 0.9
        self.num_tracked_lms = 0
        self.curr_kf = None
        self.map_cleaner = cslam.LocalMapCleaner(guided_matcher) #, camera)
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
            self.n_landmarks += 1
        
        # for lm_id in self.graph[self.init_frame.im_name]:
        #     print("lm_id: ", lm_id)
        #     e = self.graph.get_edge_data(lm_id, self.init_frame.im_name)
        #     print("e", e)
        #     print(self.graph.node[lm_id]['data'])


        print("create_init_map: len(local_landmarks): ", len(self.local_landmarks), n_lm_added)
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
        num_keyfrms = len(self.c_keyframes)
        # const unsigned int min_obs_thr = (3 <= num_keyfrms) ? 3 : 2;
        # const auto num_reliable_lms = ref_keyfrm.get_num_tracked_landmarks(min_obs_thr);
        min_obs_thr = 3 if (3 <= num_keyfrms) else 2
        last_kf = self.c_keyframes[-1]
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


        # if self.n_keyframes % 15 == 0:
        self.paint_reconstruction()
        self.save_reconstruction(frame.im_name+"bef")

        # lms_clean = new_kf.ckf.update_lms_after_kf_insert() # == store_new_keyframe
        self.map_cleaner.update_lms_after_kf_insert(new_kf.ckf)
        self.map_cleaner.remove_redundant_lms(new_kf.kf_id)
        self.create_new_landmarks()
        # if self.n_keyframes % 15 == 0:
        self.paint_reconstruction()
        self.save_reconstruction(frame.im_name+"aft")
        self.update_new_keyframe()
        self.local_bundle_adjustment()        
        self.remove_redundant_kfs()
        

    
    def update_new_keyframe(self):
        """ update new keyframe
        detect and resolve the duplication of the landmarks observed in the current frame
        """      
        # again, check the last 10 frames
        fuse_kfs = self.c_keyframes[-10:-1]
        self.map_cleaner.fuse_landmark_duplication(self.curr_kf.ckf, fuse_kfs)
        self.map_cleaner.update_new_keyframe(self.curr_kf.ckf)
        

    def local_bundle_adjustment(self):
        """ TODO: Build optimization problem directly from C++"""
        if self.n_keyframes <= 2:
            return
        return
        ba = csfm.BundleAdjuster()
        
        for camera in self.reconstruction.cameras.values():
            reconstruction._add_camera_to_bundle(ba, camera, camera, constant=True)
        # Find "earliest" KF seen by the current map! 
        n_kfs_fixed = 2
        n_kfs_optimize = 10
        # First, create a list of all frames and fix all but the newest N
        # kf_constant = self.fix_keyframes(n_kfs_fixed, n_kfs_optimize)    
        # pass
        # Add new landmarks to optimize
        # local_keyframes = self.c_keyframes[-n_kfs_optimize: -1]

        # (1), find all the kfs that see landmarks in the current frame and let's
        #       call them local keyframes
        # (2) find all the landmarks seen in local keyframes
        # (3) find all the keyframes containing the landmarks but set the ones
        #     not in local keyframes constant 
        # correct local keyframes of the current keyframe
        kf_added = {}



        local_kfs_idx = self.curr_kf.ckf.compute_local_keyframes()
        local_kfs = []
        for kf_id in local_kfs_idx:
            local_kfs.append(self.c_keyframes[kf_id])
            # add them directly to BA problem
            shot = self.reconstruction.shots[kf_id]
            r = shot.pose.rotation
            t = shot.pose.translation
            ba.add_shot(shot.id, shot.camera.id, r, t, kf_id == 0)
            kf_added[str(kf_id)] = True
        
        lm_added = {}
        # correct local landmarks seen in local keyframes
        for kf in local_kfs:
            for lm_id in self.graph[kf.im_name]:
                lm_node = self.graph(kf.im_name, str(lm_id))
                lm = lm_node['data']
                if lm.clm:
                    # add it
                    if not lm_added[str(lm_id)]:
                        # add lm
                        point = self.reconstruction.points[lm_id]
                        ba.add_point(point.id, point.coordinates, False)
                        lm_added[str(lm_id)] = True
                    
                     # add observation
                    scale = self.graph[kf_id][lm_id]['feature_scale']
                    pt = self.graph[kf_id][lm_id]['feature']
                    ba.add_point_projection_observation(kf_id, lm_id, pt[0], pt[1], scale)

        # fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
        #Now, check if all the frames were added
        # for lm_id in lm_added.keys():
        #     for kf_id

    def remove_redundant_kfs(self):
        """TODO!!"""
        pass

    def create_new_landmarks(self):
        """Creates a new landmarks with using the newly added KF
        """
        # Again, just take the last 10 frames
        # but not the current one!
        local_keyframes = self.c_keyframes[-10:-1]
        new_kf = self.c_keyframes[-1]
        new_kf_py = self.keyframes[-1]
        new_im = self.data.load_image(new_kf_py.im_name)

        py_kfs = self.keyframes[-10:-1]
        new_cam_center = new_kf.get_cam_center()
        new_Tcw = np.linalg.inv(new_kf.get_pose())
        new_R = new_Tcw[0:3, 0:3]
        new_t = new_Tcw[0:3, 3]
        print("local_kf: ", len(local_keyframes))
        # TODO! check new_kf pose
        for (old_kf, old_kf_py) in zip(local_keyframes, py_kfs):
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
            old_im = self.data.load_image(old_kf_py.im_name)
            slam_debug.disable_debug = False
            slam_debug.visualize_matches_pts(new_kf.getKptsPy(), old_kf.getKptsPy(), np.array(matches), new_im, old_im, do_show=True, is_normalized=False, title=old_kf_py.im_name)
            slam_debug.disable_debug = False
            print("kf1 before: ", new_kf_py.ckf.get_num_tracked_lms(2))    
            print("kf2 before: ", old_kf_py.ckf.get_num_tracked_lms(2))   
            self.triangulate_from_two_kfs(new_kf_py, old_kf_py, matches)
            print("kf1: ", new_kf_py.ckf.get_num_tracked_lms(2))    
            print("kf2: ", old_kf_py.ckf.get_num_tracked_lms(2))   

    def triangulate_from_two_kfs(self, new_kf_py: Keyframe, old_kf_py: Keyframe, matches):
        # TODO: try without tracks graph
        frame1 = new_kf_py.im_name
        frame2 = old_kf_py.im_name
        # create the graph
        tracks_graph = nx.Graph()
        tracks_graph.add_node(str(frame1), bipartite=0)
        tracks_graph.add_node(str(frame2), bipartite=0)
        f_processed = defaultdict(int)
        p1, _, _ = features.normalize_features(new_kf_py.ckf.getKptsPy(),None,None,self.camera[1].width,self.camera[1].height)
        p2, _, _ = features.normalize_features(old_kf_py.ckf.getKptsPy(),None,None,self.camera[1].width,self.camera[1].height)
        c1 = new_kf_py.colors
        c2 = old_kf_py.colors

        for (track_id, (f1_id, f2_id)) in enumerate(matches):
            # this checks whether the current kf was matched
            # to one of the landmarks.
            # if f2 is already in a lm
            f_processed[f1_id] += 1
            if f_processed[f1_id] > 1:
                print("double add!!")
                exit()
            x, y, s = p2[f2_id, 0:3]
            r, g, b = c2[f2_id, :]
            tracks_graph.add_node(str(track_id), bipartite=1)
            tracks_graph.add_edge(str(frame2),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f2_id),
                                  feature_color=(float(r), float(g), float(b)))

            x, y, s = p1[f1_id, 0:3]
            r, g, b = c1[f1_id, :]
            tracks_graph.add_edge(str(frame1),
                                  str(track_id),
                                  feature=(float(x), float(y)),
                                  feature_scale=float(s),
                                  feature_id=int(f1_id),
                                  feature_color=(float(r), float(g), float(b)))
        # chrono.lap("track_graph")
        cameras = self.data.load_camera_models()
        camera = next(iter(cameras.values()))
        rec_tri = types.Reconstruction()
        rec_tri.reference = self.data.load_reference()
        rec_tri.cameras = cameras

        shot1 = types.Shot()
        shot1.id = frame1
        shot1.camera = camera
        shot1.pose = new_kf_py.world_pose
        shot1.metadata = reconstruction.get_image_metadata(self.data, frame1)
        rec_tri.add_shot(shot1)

        shot2 = types.Shot()
        shot2.id = frame2
        shot2.camera = camera
        shot2.pose = old_kf_py.world_pose
        shot2.metadata = reconstruction.get_image_metadata(self.data, frame2)
        rec_tri.add_shot(shot2)

        graph_inliers = nx.Graph()
        # chrono.lap("ba setup")
        np_before = len(rec_tri.points)
        reconstruction.triangulate_shot_features(tracks_graph, graph_inliers,
                                                 rec_tri, frame1,
                                                 self.data.config)
        np_after = len(rec_tri.points)
        print("Successfully triangulated {} out of {} points.".format(np_after, np_before))
        # chrono.lap("triangulateion")
        edges1 = graph_inliers.edges(frame1)
        points = rec_tri.points
        points3D = np.zeros((len(points), 3))
        for idx, pt3D in enumerate(points.values()):
            points3D[idx, :] = pt3D.coordinates
        
        slam_debug.reproject_landmarks(points3D, None, new_kf_py.world_pose, self.data.load_image(new_kf_py.im_name), self.camera[1], do_show=False)
        slam_debug.reproject_landmarks(points3D, None, old_kf_py.world_pose, self.data.load_image(old_kf_py.im_name), self.camera[1], do_show=True)
        kf1 = new_kf_py.ckf
        kf2 = old_kf_py.ckf
        
        # Add to graph -> or better just create clm
        for _, gi_lm_id in graph_inliers.edges(frame1):
            # TODO: Write something like create_landmark
            pos_w = rec_tri.points[gi_lm_id].coordinates
            lm_id = str(self.current_lm_id)
            
            lm = Landmark(lm_id)
            clm = cslam.Landmark(int(lm_id), kf2, pos_w)
            print("Creating landmark:", lm_id, pos_w)
            # print("Creating new lm: ", lm_id)
            lm.first_kf_id = old_kf_py.kf_id
            lm.clm = clm
            self.n_landmarks += 1
            self.current_lm_id += 1
            # This is essentially the same as adding it to the graph
            # self.add_landmark(lm)
            self.graph.add_node(lm_id, data=lm)
            # Now, relate the gi_lm_id to the actual feature_id
            e1 = graph_inliers.get_edge_data(frame1, gi_lm_id)
            e2 = graph_inliers.get_edge_data(frame2, gi_lm_id)
            
            self.graph.add_edges_from([(frame1, str(lm_id), e1)])
            self.graph.add_edges_from([(frame2, str(lm_id), e2)])
            # connect landmark -> kf
            clm.add_observation(kf1, f1_id)
            print("kf1", kf1, kf1.im_name, clm.is_observable_in_kf(kf1), clm.is_observable_in_kf(kf2))
            clm.add_observation(kf2, f2_id)
            print("kf2", kf2, kf2.im_name, clm.is_observable_in_kf(kf2))
            # connect kf -> landmark in the graph
            kf1.add_landmark(clm, f1_id)
            print("add lm kf1")
            kf2.add_landmark(clm, f2_id)
            print("add lm kf2")
            clm.compute_descriptor()
            print("cmp desc")
            clm.update_normal_and_depth()
            print("add update_normal_and_depth kf1")
            self.last_frame.cframe.add_landmark(clm, f1_id)
            self.map_cleaner.add_landmark(clm)
            # TODO: check if this the correct frame

            # We also have to add the points to the reconstruction
            point = types.Point()
            point.id = str(lm_id)
            point.coordinates = rec_tri.points[gi_lm_id].coordinates
            self.reconstruction.add_point(point)
            # self.local_landmarks.append(lm.lm_id)
            # self.add_fresh_landmark(lm.lm_id)
            # only +1 because already init with 1
            # lm.num_observed += 1
            # lm.num_observable += 1
        print("kf1 in func: ", new_kf_py.ckf.get_num_tracked_lms(2), kf1.get_num_tracked_lms(2))    
        print("kf2 in func: ", old_kf_py.ckf.get_num_tracked_lms(2), kf2.get_num_tracked_lms(2))    

    

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

        self.keyframes.append(kf)
        
        self.c_keyframes.append(kf.ckf)
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


# def create_init_map2(self, graph_inliers, rec_init,
#                         init_frame: Frame, curr_frame: Frame,
#                         init_pdc=None, other_pdc=None):
#         """The graph contains the KFs/shots and landmarks.
#         Edges are connections between keyframes and landmarks and
#         basically "observations"
#         """

#         lm_id = 0
#         lm_c = ctypes.Landmark(lm_id)
#         kf1 = ctypes.KeyFrame(0, init_frame.cframe)
#         kf2 = ctypes.KeyFrame(1, curr_frame.cframe)

#         self.graph = graph_inliers
#         self.reconstruction = rec_init
#         init_frame.frame_id = 0
#         # Create keyframes
#         self.init_frame = Keyframe(init_frame, self.data, 0)
#         self.init_frame.world_pose = \
#             rec_init.shots[init_frame.im_name].pose
#         curr_frame.frame_id = 1
#         curr_kf = Keyframe(curr_frame, self.data, 1)
#         curr_kf.world_pose = rec_init.shots[curr_frame.im_name].pose

#          # Add to data and covisibility
#         self.add_keyframe(self.init_frame)
#         self.add_keyframe(curr_kf)
#         self.n_frames = 2
#         max_lm = 0  # find the highest lm id
#         n_lm_added = 0

#         # Add landmark objects to nodes
#         for lm_id in self.graph[self.init_frame.im_name]:
#             lm = Landmark(int(lm_id))
#             lm.num_observable = 2
#             lm.num_observed = 2
#             lm.first_kf_id = self.init_frame.kf_id
#             self.graph.add_node(lm_id, data=lm)

#             int_id = int(lm_id)
#             if int_id > max_lm:
#                 max_lm = int_id
#             lm.compute_descriptor(curr_kf, self.graph) # update descriptor
#             pos_w = rec_init.points[str(lm_id)].coordinates # update normal
#             lm.update_normal_and_depth(pos_w, self.graph)
#             self.local_landmarks.append(lm_id)
#             n_lm_added += 1

#         logger.debug("create_init_map: len(local_landmarks): %, %".
#                      format(len(self.local_landmarks), n_lm_added))
#         self.current_lm_id = max_lm + 1
#         # TODO: think about the local landmarks