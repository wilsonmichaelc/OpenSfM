#include <algorithm>
#include "point.h"

void
Pose::setPose(const Pose& pose)
{
  worldToCam_ = pose.WorldToCamera();
  camToWorld_ = pose.CameraToWorld();
}

bool 
Point::isObservedInShot(Shot* shot) const
{
  return observations_.count(shot);
}
void 
Point::addObservation(Shot* shot, const FeatureId feat_id)
{
  observations_.emplace(shot, feat_id);
}
void
Point::hasObservations() const
{
  return !observations_.empty();
}
void
Point::removeObservation(Shot* shot)
{
  observations_.erase(shot);
}

Point::Point(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name):
  id_(point_id), global_pos_(global_pos), point_name_(name)
{

}


Shot::Shot(const ShotId shot_id, const std::string& name, const camera* camera, const Pose& pose):
            id_(shot_id), image_name(name), camera_(camera), pose_(pose)
{
  
}

void
Shot::removePointObservation(const FeatureId id)
{
  points_.at(id) = nullptr;
}

size_t
Shot::computeNumValidPoints() const
{
  return points_.size() - std::count(points_.cbegin(), points_.cend(), nullptr);
}

Shot*
ReconstructionManager::CreateShot(const ShotId shot_id, const CameraId camera_id, const Pose& pose, const std::string& name)
{
  const auto* shot_cam = cameras_.at(camera_id);
  //Create a unique ptr and move it into the object?
  auto new_shot = std::make_unique<Shot>(shot_id, shot_cam, pose, name);
  auto new_shot_raw = new_shot.get();
  shots_.emplace(shot_id, std::move(new_shot));
  if (!name.empty())
    shot_names_.emplace(name, shot_id);
  return new_shot_raw;
}

bool
ReconstructionManager::UpdateShotPose(const ShotId shot_id, const Pose& pose)
{
  auto shot = shots_.find(shot_id)
  if (shot == shots_.end())
    return false;
  
  shot.setPose(pose);
  return true;
}

Point*
ReconstructionManager::CreatePoint(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name = "")
{
  //Create a unique ptr and move it into the object?
  auto new_point = std::make_unique<Point>(point_id, global_pos, name);
  auto new_point_raw = new_point.get();
  shots_.emplace(point_id, std::move(new_shot));
  if (!name.empty())
    points_.emplace(name, point_id);
  return new_point_raw;
}

bool
ReconstructionManager::UpdatePoint(const PointId point_id, const Eigen::Vector3d& global_pos)
{
  auto point = points.find(point_id)
  if (point == shots_.end())
    return false;
  
  point.SetPointInGlobal(pose);
  return true;
}

bool 
AddObservation(const Shot* shot, const Point* point, const FeatureId feat_id)
{
  // 1) Check that shot and point exist
  if (shots_.at(shot->id_) == shots_.end() || points_.at(point->id_) == points_.end())
    return false
  
  // 2) add observations
  shot->addPointObservation(point, feat_id);
  point->addObservation(shot, feat_id);

  return true;
}

bool
RemoveObservation(const Shot* shot, const Point* point, const FeatureId feat_id)
{
  // 1) Check that shot and point exist
  if (shots_.at(shot->id_) == shots_.end() || points_.at(point->id_) == points_.end())
    return false

  // 2) remove observations
  shot->removePointObservation(feat_id);
  point->removeObservation(shot);
  return true;
}
