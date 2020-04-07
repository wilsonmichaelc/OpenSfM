import numpy as np
from opensfm import pymap



def test_create_shot_cameras(n_cams, map_mgn, cam_model):
  cams = []
  for i in range(0, n_cams):
    cam = map_mgn.create_shot_camera(i, cam_model, "cam"+str(i))
    print(cam)
    cams.append(cam)
  return cams

def test_delete_shot_cameras(cams, map_mgn):
  print("Deleting ", len(cams), " cameras") 
  for cam in cams:
    print("Delete cam: ", cam.camera_name_, " with id: ", cam.id_)
    map_mgn.remove_shot_camera(cam.id_)

  print("Number of cameras after delete: ", map_mgn.number_of_cameras())


def test_pose():
  pose = pymap.Pose()
  print("Default pose: \n cam_to_world \n{}, world_to_cam \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))
  
  # TODO: Set with actual transformation matrix!
  pose.set_from_world_to_cam(np.array([[1,0,0,30],[0,1,0,20],[0,0,1,10],[0,0,0,1]]))
  print("Modify pose with set_from_world_to_cam: \n cam_to_world: \n{} \n world_to_cam: \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))
  pose.set_from_cam_to_world(np.array([[1,0,0,30],[0,1,0,20],[0,0,1,10],[0,0,0,1]]))
  print("Modify pose with set_from_cam_to_world: \n cam_to_world: \n{} \n world_to_cam: \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))

def test_create_shots(map_mgn, n_shots, cam):
  shots = []
  pose = pymap.Pose()
  for shot_id in range(n_shots):
    shots.append(map_mgn.create_shot(shot_id, cam, pose, "shot"+str(shot_id)))
    print("created: ", shots[-1].id_, ": ", shots[-1].name_, "obj: ", shots[-1])
  
  #-----------Try to create everything again ------------
  for shot_id in range(n_shots):
    shots.append(map_mgn.create_shot(shot_id, cam, pose, "shot"+str(shot_id)))
    print("created: ", shots[-1].id_, ": ", shots[-1].name_, "obj: ", shots[-1])
  return shots


def test_remove_shots(map_mgn):
  print("Number of Shots ", map_mgn.number_of_shots())
  shot_ids = [0, 5, 6]
  for shot_id in shot_ids:
      map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
        format(map_mgn.number_of_shots(), shot_ids))

  #------------DELETE the same multiple times--------------
  shot_ids = [1,1,1]
  for shot_id in shot_ids:
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
    format(map_mgn.number_of_shots(), shot_ids))

  #------------DELETE ALREADY DELETED--------------
  shot_ids = [5,6,8]  
  for shot_id in shot_ids:
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
    format(map_mgn.number_of_shots(), shot_ids))

  #------------DELETE ALL--------------
  all_shots = map_mgn.get_all_shots()
  for shot_id in all_shots.keys():
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting all".
    format(map_mgn.number_of_shots()))

  #------------DELETE ALL AGAIN--------------
  all_shots = map_mgn.get_all_shots()
  for shot_id in all_shots.keys():
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
    format(map_mgn.number_of_shots(), shot_ids))

def print_all_shots(map_mgn):
  all_shots = map_mgn.get_all_shots()
  print("Number of Shots ", len(all_shots))
  for shot in all_shots.values():
    print("Shots in map mgn: ", shot.id_)

def test_create_landmarks(map_mgn, n_lms):
  lms = []
  
  for lm_id in range(n_lms):
    lms.append(map_mgn.create_landmark(lm_id, np.random.rand(3,1), "lm"+str(lm_id)))
    lm = map_mgn.create_landmark(lm_id, np.random.rand(3,1), "lm"+str(lm_id))
    if (lm != lms[-1]):
      print("Double creation!")
      exit()
  print("Test create landmarks passed")

def test_remove_landmarks(map_mgn):
  lm_to_delete = np.array(np.random.rand(200,1)*1000, dtype=np.int)
  print("Landmarks before: ", map_mgn.number_of_landmarks(), len(np.unique(lm_to_delete)), np.max(lm_to_delete))
  for lm in lm_to_delete:
    map_mgn.remove_landmark(lm)
  print("Landmarks after: ", map_mgn.number_of_landmarks())

def test_larger_problem(map_mgn):
  # Create 2 cameras
  cam1 = pymap.Camera()
  cam2 = pymap.Camera()
  # Create 2 shot cameras
  shot_cam1 = map_mgn.create_shot_camera(0, cam1, "shot_cam1")
  shot_cam2 = map_mgn.create_shot_camera(1, cam2, "shot_cam2")
  # Create 10 shots, 5 with each shot camera
  shots = []
  for shot_id in range(0,5):
    shots.append(map_mgn.create_shot(shot_id, shot_cam1))
  for shot_id in range(5, 10):
    shots.append(map_mgn.create_shot(shot_id, 1))
    # shots[-1].init_keypts_and_descriptors(100)
  for shot in shots:
    shot.init_keypts_and_descriptors(100)
  # Create 200 landmarks
  landmarks = []
  for lm_id in range(0, 200):
    lm = map_mgn.create_landmark(lm_id,np.random.rand(3,1),"lm"+str(lm_id))
    print("Create lm: ", lm_id, lm, len(landmarks))
    landmarks.append(lm)
  
  # assign 100 to each shot (observations)
  for shot in shots: 
    lm_obs = np.asarray(np.random.rand(100)*1000%200, dtype=np.int)
    feat_obs = np.asarray(np.random.rand(100)*1000%100, dtype=np.int)
    # print("Before {} observations to shot {}".format(shot.compute_num_valid_pts(), shot.id_))
    for f_idx, lm_id in zip(feat_obs, lm_obs):
      # print(f_idx, lm_id, shot.number_of_keypoints())
      map_mgn.add_observation(shot, landmarks[lm_id],f_idx)
    print("Added {} observations to shot {}".format(shot.compute_num_valid_pts(), shot.id_))
    if (len(np.unique(feat_obs)) != shot.compute_num_valid_pts()):
      print("Error double add!" )
      return

  for lm in landmarks:
    print("lm: ", lm.id_, " has ", lm.number_of_observations())
    if (lm.has_observations()):
      shots = lm.get_observations()
      shot = list(shots.keys())[0]
      #TODO: print all the observations of shot
      map_mgn.remove_shot(shot.id_)
    print("lm: ", lm.id_, " has ", lm.number_of_observations(), "after delete!")

  # Remove 50 landmarks
  # lm_obs = np.unique(np.asarray(np.random.rand(50)*1000%200, dtype=np.int))
  # for lm_id in lm_obs:
    # map_mgn.remove_landmark(lm_id)
  map_mgn.remove_landmark(0)
  


  for shot in shots:
      print("{} observations in shot {}".format(shot.compute_num_valid_pts(), shot.id_))

  # for shot in shots: 
    # feat_obs = np.asarray(np.random.rand(50)*1000%200, dtype=np.int)
    # print("Removing from {} with a tot of {} observations:".format(shot.id_, shot.compute_num_valid_pts()))
    # for feat_id in feat_obs:
      # map_mgn.remove_observation(shot, feat_id)
    # print("Removed from {} with a tot of {} observations:".format(shot.id_, shot.compute_num_valid_pts()))

  # Just try to delete it from all the shots
  # To test what happens if we delete it from a shot without the observation


  map_mgn.add_observation(shots[-1], landmarks[-1], 0)



  


# Map Manager
map_mgn = pymap.Manager()
cam_model = pymap.Camera()

# Test the cameras
cams = test_create_shot_cameras(10, map_mgn, cam_model)
print("Created {} cameras ".format( len(cams)))
print(cam_model)
test_delete_shot_cameras(cams, map_mgn)

# Test the pose
test_pose()

cam = map_mgn.create_shot_camera(0, cam_model, "cam"+str(0))

# Test the shots
shots = test_create_shots(map_mgn,10,cam)
test_remove_shots(map_mgn)

#Test the landmarks
landmarks = test_create_landmarks(map_mgn, 1000)
test_remove_landmarks(map_mgn)


#Now create a large problem
test_larger_problem(map_mgn)


