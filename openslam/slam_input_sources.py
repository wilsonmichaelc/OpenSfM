import cv2


def video_source(object):

    def __init__(self, video_path, skip_frames=0):
        self.video_stream = cv2.VideoCapture(video_path)
        if (self.video_stream and skip_frames > 0):
            cv2.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

    def get_next_frame(self, frame):
        """Return the next frame in the video stream"""
        return self.video_stream.read(frame)


def image_source(self, object):

    def __init__(self, dataset):
        self.folder_path = dataset.data_path
        self.sorted_image_list = sorted(dataset.image_list)
        """Creates a list of imfile names to read from"""
        self.curr_frame_n = 0

    def get_next_frame(self, frame):
        """Return the next frame in the file list"""
        if (self.curr_frame_n < len(self.image_list)):
            frame = cv2.imread(self.image_list[self.curr_frame_n])
        else:
            frame = None
        return frame is not None
