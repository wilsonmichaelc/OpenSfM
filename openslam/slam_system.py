from opensfm import dataset
from opensfm import reconstruction
from slam_initializer import SlamInitializer
from slam_mapper import SlamMapper
from slam_tracker import SlamTracker
from slam_types import Frame
from slam_feature_extractor import SlamFeatureExtractor
import slam_debug
import slam_utils
import slam_config
import logging
import orb_extractor
import guided_matching
import numpy as np
# from opensfm import csfm
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SlamSystem(object):

    def __init__(self, args):
        self.data = dataset.DataSet(args.dataset)
        cameras = self.data.load_camera_models()
        self.config = self.data.config
        self.camera = next(iter(cameras.items()))
        self.config_slam = slam_config.default_config()
        self.slam_mapper = SlamMapper(self.data, self.config_slam, self.camera)
        self.slam_tracker = SlamTracker(self.data, self.camera)

        self.feature_extractor = SlamFeatureExtractor(self.config_slam)

        # TODO: Move to separate cl ass
        self.orb_extractor = orb_extractor.orb_extractor(
            self.config_slam['feat_max_number'],
            self.config_slam['feat_scale'],
            self.config_slam['feat_pyr_levels'],
            self.config_slam['feat_fast_ini_th'],
            self.config_slam['feat_fast_min_th']
        )

        corner_pts = np.array([[0, 0], # left top
                               [self.camera[1].width, 0], # right top
                               [0, self.camera[1].height], # left bottom
                               [self.camera[1].width, self.camera[1].height]]) # right bottom
        
        corners = self.camera[1].undistort_many(corner_pts).reshape((4, 2))
        print(corners)
        bounds = np.array([np.min((corners[0, 0], corners[2, 0])),
                           np.max((corners[1, 0], corners[3, 0])),
                           np.min((corners[0, 1], corners[2, 1])),
                           np.max((corners[1, 1], corners[3, 1]))])
        # ])
        print(bounds)
        inv_cell_w = self.config_slam['grid_n_cols']/(bounds[1]-bounds[0])
        inv_cell_h = self.config_slam['grid_n_rows']/(bounds[3]-bounds[2])
        self.grid_params =\
            guided_matching.\
            GridParameters(self.config_slam['grid_n_cols'],
                           self.config_slam['grid_n_rows'],
                           bounds[0], bounds[2], inv_cell_w, inv_cell_h)
        
        self.slam_init = SlamInitializer(self.data, self.camera, self.grid_params)
        self.system_initialized = False
        

    def process_frame(self, frame):
        """Process one single frame.
        Step 1: Extract multi-scale features
        Step 2: Init or track frame
        """
        # orb_detector.detect()
        keypts = list()
        mask = np.array([], dtype=np.uint8)
        chrono = reconstruction.Chronometer()
        # Step 1: Detect SIFT/ORB features in first image
        # ORB
        kpt, desc = self.orb_extractor.extract_orb_py(frame.image, mask)
        # f = orb_extractor.Frame(10)
        # f.print_info()
        # self.orb_extractor.extract_orb_py2(frame.image, mask, f)
        # f.print_info()
        up = self.camera[1].undistort_many(kpt[:, 0:2])
        # l = guided_matching.assign_keypoints_to_grid(self.grid_params, up.reshape(-1,2))
        slam_debug.draw_obs_in_image_no_norm(kpt, frame.image)
        frame.points = kpt
        frame.undist_pts = up.reshape(-1, 2)
        frame.keypts_in_cell = guided_matching.\
            assign_keypoints_to_grid(self.grid_params, frame.undist_pts)
        frame.descriptors = desc
        frame.colors = slam_utils.extract_colors_for_pts(frame.image, kpt)
        
        chrono.lap('new_orb')
        if not self.system_initialized:
            return self.init_slam_system(frame)
        self.track_frame(frame)

    def init_slam_system(self, frame: Frame):
        """Find the initial depth estimates for the slam map"""
        print("init_slam_system: ", frame)
        data = self.data
        if self.slam_init.init_frame is None:
            self.slam_init.set_initial_frame(frame)
            self.system_initialized = False
        else:
            rec_init, graph, matches = \
                self.slam_init.initialize(frame)
            self.system_initialized = (rec_init is not None)
            if self.system_initialized:
                self.slam_mapper.create_init_map(graph, rec_init,
                                                 self.slam_init.init_frame,
                                                 frame,
                                                 self.slam_init.init_pdc,
                                                 self.slam_init.other_pdc)
        if self.system_initialized:
            # chrono.lap('init')
            # slam_debug.avg_timings.addTimes(chrono.laps_dict)
            logger.debug("Initialized system with {}".format(frame.im_name))
        else:
            logger.debug("Failed to initialize with {}".format(frame.im_name))
        return self.system_initialized


    def track_frame(self, frame: Frame):
        """ Tracks a frame
        """
        data = self.data
        logger.debug("Tracking: {}, {}".format(frame.frame_id, frame.im_name))
        # Maybe move most of the slam_mapper stuff to tracking
        self.slam_mapper.apply_landmark_replace()
        pose = self.slam_tracker.track(self.slam_mapper, frame,
                                        self.config, self.camera, data)


