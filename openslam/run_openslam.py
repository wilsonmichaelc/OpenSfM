import os.path, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from openslam import initializer
from slam_frame import Frame
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
# video_path = "/home/fschenk/data/ae_sequences/videos/flir_videos/flir-1080-4.mp4"

data = dataset.DataSet(args.dataset)
# print("data.image_files", data.image_files)
input_source = image_source(data)
# data.load_image()
frame = None


# print("data.images(): ", data.images())
# print("image_source.sorted_image_list: ", input_source.sorted_image_list)
# initializer = slam_initializer()

for im_name in sorted(data.image_list):
    ret = slam_system.track_next_frame(data, im_name)
    if ret:
        print("slam tracking")
        print(im_name)
    else:
        print("slam trying to init")

exit()
frame = input_source.get_next_frame()
n_frames = 1
while frame is not None:
    print("Number of processed frames: ", n_frames)
    #features detected
    # break
    slam_system.track_next_frame(data, frame)
    frame = input_source.get_next_frame()
    # if frame is not None:
        
    n_frames += 1
    # slam_system.track_next_frame(frame)
    # print(data.load_features('01.jpg'))
    
