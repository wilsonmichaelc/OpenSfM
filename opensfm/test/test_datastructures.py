import numpy as np
from opensfm import pymap



def test_create_shot_cameras(n_cams, map_mgn):
  cams = []
  cam_model = pymap.Camera()
  for i in range(0, n_cams):
    cams.append(map_mgn.create_camera(i, cam_model, "cam"+str(i)))
  
  print("Created {} cameras ".format( len(cams)))
  for cam in cams:
    print("Camera: ", cam.camera_name_, " with id: ", cam.id_)
  return cams

def test_delete_shot_cameras(cams, map_mgn):
  print("Deleting ", len(cams), " cameras") 
  for cam in cams:
    print("Delete cam: ", cam.camera_name_, " with id: ", cam.id_)
    map_mgn.remove_camera(cam.id_)

  print("Number of cameras after delete: ", map_mgn.number_of_cameras())


def test_pose():
  pose = pymap.Pose()
  print("Default pose: \n cam_to_world \n{}, world_to_cam \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))
  
  # TODO: Set with actual transformation matrix!
  pose.set_from_world_to_cam(np.array([[1,2,3,4],[6,7,8,9],[10,11,12,13],[14,15,16,17]]))
  print("Modify pose with set_from_world_to_cam: \n cam_to_world: \n{} \n world_to_cam: \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))
  pose.set_from_cam_to_world(np.array([[1,2,3,4],[6,7,8,9],[10,11,12,13],[14,15,16,17]]))
  print("Modify pose with set_from_cam_to_world: \n cam_to_world: \n{} \n world_to_cam: \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))

def test_shots(map_mgn):
  cam_model = pymap.Camera()
  # all shots get the same camera for now
  cam = map_mgn.create_camera(0, cam_model, "cam"+str(0))
  shot1 = map_mgn.create_shot(0, cam.id_)
  shot2 = map_mgn.create_shot(1, cam)
  pose = pymap.Pose()
  shot3 = map_mgn.create_shot(2, cam, pose)
  shot4 = map_mgn.create_shot(3, cam, pose, "shot4")
  print("shot: ", shot4.id_, ": ", shot4.shot_name_)


# Map Manager
map_mgn = pymap.Manager()

# Test the cameras
cams = test_create_shot_cameras(10, map_mgn)
test_delete_shot_cameras(cams, map_mgn)

# Test the pose
test_pose()


# Test the shots
test_shots(map_mgn)
