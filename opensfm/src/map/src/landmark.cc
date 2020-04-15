#include <algorithm>
#include <map/landmark.h>
#include <map/shot.h>
#include <slam/guided_matching.h>

namespace map
{
Landmark::Landmark(const LandmarkId lm_id, const Eigen::Vector3d& global_pos, const std::string& name):
  id_(lm_id), name_(name), global_pos_(global_pos)
{

}

double 
Landmark::ComputeDistanceFromRefFrame() const
{
  const Eigen::Vector3d cam_to_lm_vec = global_pos_ - ref_shot_->GetPose().GetOrigin();
  return cam_to_lm_vec.norm();
}

}; //namespace map
