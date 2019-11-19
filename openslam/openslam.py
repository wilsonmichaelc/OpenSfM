import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initializer import SlamInitializer
from slam_matcher import SlamMatcher
from slam_mapper import SlamMapper

# from openslam import initializer

from opensfm import dataset
from opensfm import features
from opensfm import reconstruction
from opensfm import feature_loader
# from initializer import slam_initializer
import numpy as np
from opensfm import feature_loading
import networkx as nx


class SlamSystem(object):
    
    
    def __init__(self, args):
        print("Init slam system", args)
        self.data = dataset.DataSet(args.dataset)
        cameras = self.data.load_camera_models()
        self.config = self.data.config
        self.camera = next(iter(cameras.items()))
        self.system_initialized = False
        self.system_lost = True
        self.slam_matcher = SlamMatcher(self.config)
        self.initializer = SlamInitializer(self.config, self.slam_matcher)
        self.tracked_frames = 0
        # self.feature_loader = feature_loading.FeatureLoader()
        self.image_list = sorted(self.data.image_list)
        # self.global_graph = nx.Graph()
        
        self.slam_mapper = SlamMapper(self.data, self.config)
        self.initializer.matcher = self.slam_matcher

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    # def run(self,args)
    #     print("Init slam system")
    #     data = dataset.DataSet(args.dataset)
    #     camera = data.load_camera_models[0]
    #     print("camera")
    #     self.camera = camera
    #     self.system_initialized = False
    #     self.system_lost = True

    def init_slam_system(self, data, frame):
        """Find the initial depth estimates for the slam map"""
        if self.tracked_frames == 0:
            self.initializer.set_initial_frame(data, frame)
            matches = []
        else:
            self.system_initialized, matches = self.initializer.initialize(data, frame)
        print("Returning", self.system_initialized)
        return self.system_initialized, matches

    def track_next_frame(self, data, frame): #, slam_matcher, slam_mapper):

        """Estimates the pose for the next frame"""
        if not self.system_initialized:
            success, matches = self.init_slam_system(data, frame)
            im1, im2 = self.initializer.ref_frame.im_name, frame.im_name
            print(im1, im2)
            p1, f1, _ = feature_loader.instance.load_points_features_colors(
                data, im1, masked=True)
            p2, f2, _ = feature_loader.instance.load_points_features_colors(
                data, im2, masked=True)
            threshold = data.config['five_point_algo_threshold']
            # print(self.camera)
            # print(self.camera[1].pixel_bearing_many(p1))
            R, t, inliers,_ = \
                reconstruction.two_view_reconstruction_general(p1, p2, self.camera[1], self.camera[1], threshold)
            # print("here",p1)
            # print(len(p1),len(p1(matches[:,0])))
            print("R: ", R, " t: ", t)
            print("inliers: ", inliers)
            if success:
                #initialized
                self.slam_mapper.add_new_tracks(self.initializer.ref_frame.im_name,
                                                frame.im_name, matches)
                # slam_matcher.match(self.initializer)
                # matching.match
                # system is initialized now
                # add first frame and current frame to the graph!        
                # self.global_graph.add_node(str(self.initializer.ref_frame.im_name), bipartite=0)
                # self.global_graph.add_node(str(frame.im_name), bipartite=0)
                #now add the matches between both -> where do I get the matches from?
                #for track_id 
                # tracks_graph.add_node(str(track_id), bipartite=1)
                # tracks_graph.add_edge(str(image),
                #                     str(track_id),
                #                     feature=(float(x), float(y)),
                #                     feature_scale=float(s),
                #                     feature_id=int(featureid),
                #                     feature_color=(float(r), float(g), float(b)))
                
                # TODO: Use _good_track....


        
        

        self.tracked_frames += 1
        return not self.system_lost

    def local_optimization(self):
        return True

    def global_optimization(self):
        return True

    def reset(self):
        """Reset the system"""
        return True

    def relocalize(self, frame):
        """Relocalize against the SLAM map either after tracking loss or
           alternatively if the region is sufficiently explored
        """
        return True

    def save_slam_reconstruction(self):
        """Saves the slam recontsruction in the OpenSfM format"""
        return True

    def save_slam_trajectory(self, kf_only=False):
        """Saves the trajectory file of all frames or only of KFs"""
        return True

    def detect(self, data, frame):
        image = self.image_list[self.tracked_frames]
        p_sorted, f_sorted, c_sorted = self.feature_loader.load_points_features_colors(data,image)
        if p_sorted is None or f_sorted is None or c_sorted is None:
            p_unmasked, f_unmasked, c_unmasked = features.extract_features(
                data.load_image(image), data.config)

            fmask = self.data.load_features_mask(image, p_unmasked)

            p_unsorted = p_unmasked[fmask]
            f_unsorted = f_unmasked[fmask]
            c_unsorted = c_unmasked[fmask]

            if len(p_unsorted) == 0:
                logger.warning('No features found in image {}'.format(image))
                return

            size = p_unsorted[:, 2]
            order = np.argsort(size)
            p_sorted = p_unsorted[order, :]
            f_sorted = f_unsorted[order, :]
            c_sorted = c_unsorted[order, :]
            data.save_features(image, p_sorted, f_sorted, c_sorted)

