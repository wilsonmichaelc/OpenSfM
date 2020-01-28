from opensfm import reconstruction
import logging

logger = logging.getLogger(__name__)


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
