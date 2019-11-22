from opensfm import feature_loader
from opensfm import features
import logging

logger = logging.getLogger(__name__)


class Frame(object):

    def __init__(self, name):
        self.im_name = name

    def load_points_features_clr(self, data):
        # image = self.image_list[self.tracked_frames]
        image = self.im_name
        has_features = data.features_exist(image)
        if has_features:
            return feature_loader.instance.load_points_features_colors(data, image)
        # p_sorted, f_sorted, c_sorted = feature_loader.load_points_features_colors(data, image)
        # if p_sorted is None or f_sorted is None or c_sorted is None:
        # if not has_features:
        p_unmasked, f_unmasked, c_unmasked = features.extract_features(
            data.load_image(image), data.config)

        fmask = data.load_features_mask(image, p_unmasked)

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
        return p_sorted, f_sorted, c_sorted
        # else:
        

