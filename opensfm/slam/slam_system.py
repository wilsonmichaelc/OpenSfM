from opensfm import pymap
from opensfm import dataset
from opensfm import pyslam
import numpy as np
import slam_config
class SlamSystem(object):
    def __init__(self, args):
        self.data = dataset.DataSet(args.dataset)
        self.config = self.data.config
        self.config_slam = slam_config.default_config()
        self.camera = next(iter(self.data.load_camera_models().items()))

        self.map = pymap.Map()
        # Create the camera model
        self.cam = pymap.Camera()
        # Create the matching shot camera
        self.shot_cam = self.map.create_shot_camera(0, self.cam)

        self.extractor = pyslam.OrbExtractor(
            self.config_slam['feat_max_number'],
            self.config_slam['feat_scale'],
            self.config_slam['feat_pyr_levels'],
            self.config_slam['feat_fast_ini_th'],
            self.config_slam['feat_fast_min_th']
        )

    def process_frame(self, im_name, gray_scale_img):
        shot_id = self.map.next_unique_shot_id()
        curr_shot: pymap.Shot = self.map.create_shot(shot_id, self.shot_cam, im_name)
        print("Created shot: ", curr_shot.name, curr_shot.id)
        self.extractor.extract_to_shot(curr_shot, gray_scale_img, np.array([]))
        print("Extracted: ", curr_shot.number_of_keypoints())
        pass
