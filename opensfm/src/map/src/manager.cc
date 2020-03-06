#include <map/manager.h>
#include <map/landmark.h>
#include <map/camera.h>
#include <map/shot.h>
#include <map/pose.h>

namespace map
{

void 
Manager::AddObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const
{
  shot->AddPointObservation(lm, feat_id);
  lm->AddObservation(shot, feat_id);
}

void
Manager::RemoveObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const
{
  shot->RemoveLandmarkObservation(feat_id);
  lm->RemoveObservation(shot);
}

Shot*
Manager::CreateShot(const ShotId shot_id, const CameraId camera_id, const Pose& pose, const std::string& name)
{
  return CreateShot(shot_id, *cameras_.at(camera_id), pose, name);
}

Shot*
Manager::CreateShot(const ShotId shot_id, const ShotCamera& shot_cam, const Pose& pose, const std::string& name)
{
  auto it = shots_.emplace(shot_id, std::make_unique<Shot>(shot_id, shot_cam, pose, name));
  
  // Insert failed
  if (!it.second)
  {
    return nullptr;

  }

  if (!name.empty())
  {  
    shot_names_.emplace(name, shot_id);
  }
  return it.first->second.get();
}

void
Manager::UpdateShotPose(const ShotId shot_id, const Pose& pose)
{
  shots_.at(shot_id)->SetPose(pose);
}

void 
Manager::RemoveShot(const ShotId shot_id)
{
    //1) Find the point
  const auto& shot_it = shots_.find(shot_id);
  if (shot_it != shots_.end())
  {
    const auto& shot = shot_it->second;
    //2) Remove it from all the points
    for (const auto& point : shot->GetPoints())
    {
      if (point != nullptr)
      {
        point->RemoveObservation(shot.get());
      }
    }

    //3) Remove from shot_names
    shot_names_.erase(shot->shot_name_);

    //4) Remove from shots
    shots_.erase(shot_it);
  }
}

Landmark*
Manager::CreateLandmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos, const std::string& name)
{

  auto it = landmarks_.emplace(lm_id, std::make_unique<Landmark>(lm_id, global_pos, name));
  
  // Insert failed
  if (!it.second)
  {
    return nullptr;
  }

  if (!name.empty())
  {  
    landmark_names_.emplace(name, lm_id);
  }
  
  return it.first->second.get(); //the raw pointer
}

void
Manager::UpdateLandmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos)
{
  landmarks_.at(lm_id)->SetGlobalPos(global_pos);
}

void 
Manager::RemoveLandmark(const LandmarkId lm_id)
{
  //1) Find the point
  const auto& point_it = landmarks_.find(lm_id);
  if (point_it != landmarks_.end())
  {
    const auto& point = point_it->second;
    //2) Remove all its observation
    const auto& observations = point->GetObservations();
    for (const auto& obs : observations)
    {
      Shot* shot = obs.first;
      const auto feat_id = obs.second;
      shot->RemoveLandmarkObservation(feat_id);
    }

    //3) Remove from point_names
    landmark_names_.erase(point->point_name_);

    //4) Remove from points
    landmarks_.erase(point_it);
  }
}

ShotCamera* 
Manager::CreateShotCamera(const CameraId cam_id, const Camera& camera, const std::string& name)
{
  // const auto& shot_cam = cameras_.at(camera_id);
  auto it = cameras_.emplace(cam_id, std::make_unique<ShotCamera>(camera, cam_id, name));
  
  // Insert failed
  if (!it.second)
  {
    return nullptr;

  }

  if (!name.empty())
  {  
    camera_names_.emplace(name, cam_id);
  }
  return it.first->second.get();
}


void
Manager::RemoveShotCamera(const CameraId cam_id)
{
  const auto& cam_it = cameras_.find(cam_id);
  if (cam_it != cameras_.end())
  {
    cameras_.erase(cam_it);
  }
}

};
