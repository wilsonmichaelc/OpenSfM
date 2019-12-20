import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from openslam import initializer
from slam_types import Frame
from openslam import SlamSystem

from slam_input_sources import video_source
from slam_input_sources import image_source

import argparse
import logging
import cv2
from opensfm import dataset
from opensfm import features
# as input

logger = logging.getLogger(__name__)

# Create the top-level parser
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='dataset to process')
args = parser.parse_args()
slam_system = SlamSystem(args)
data = dataset.DataSet(args.dataset)
input_source = image_source(data)
for im_name in sorted(data.image_list):
    # Create frame with name and unique id
    frame = Frame(im_name, slam_system.slam_mapper.n_frames)
    print("frame: ", frame.im_name, frame.frame_id)
    ret = slam_system.track_next_frame(data, frame)
    if ret:
        print("slam tracking")
        print(im_name)
    else:
        print("slam trying to init")

slam_system.slam_mapper.paint_reconstruction(data)
slam_system.slam_mapper.save_reconstruction(data, frame.im_name+"aft")
