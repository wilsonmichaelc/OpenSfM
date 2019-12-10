import matplotlib.pyplot as plt
from opensfm import features
import logging
logger = logging.getLogger(__name__)


def reproject_landmarks(points3D, observations, pose_world_to_cam,
                        image, camera, data, do_show=True):
    """Draw observations and reprojects observations into image"""
    if points3D is None or observations is None:
        return
    if len(points3D) == 0 or len(observations) == 0:
        return
    camera_point = pose_world_to_cam.transform_many(points3D)
    points2D = camera.project_many(camera_point)
    fig, ax = plt.subplots(1)
    im = data.load_image(image)
    print("Show image ", image)
    h1, w1, c = im.shape
    pt = features.denormalized_image_coordinates(points2D, w1, h1)
    obs = features.denormalized_image_coordinates(observations, w1, h1)
    ax.imshow(im)
    ax.scatter(pt[:, 0], pt[:, 1], c=[[1, 0, 0]])
    ax.scatter(obs[:, 0], obs[:, 1], c=[[0, 1, 0]])
    if do_show:
        plt.show()


def draw_observations_in_image(observations, image, data, do_show=True):
    """Draws observations into image"""
    if observations is None:
        return
    if len(observations) == 0:
        return
    fig, ax = plt.subplots(1)
    im = data.load_image(image)
    h1, w1, c = im.shape
    obs = features.denormalized_image_coordinates(observations, w1, h1)
    ax.imshow(im)
    ax.scatter(obs[:, 0], obs[:, 1], c=[[0, 1, 0]])
    if do_show:
        plt.show()
