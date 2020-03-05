#include "reconstruction/shot.h"
#include "reconstruction/camera.h"
#include "reconstruction/point.h"
#include <algorithm>
namespace reconstruction
{
Shot::Shot(const ShotId shot_id, const ShotCamera& camera, const Pose& pose, const std::string& name):
            id_(shot_id), shot_name_(name), camera_(camera), pose_(pose)
{
  
}

void
Shot::RemovePointObservation(const FeatureId id)
{
  points_.at(id) = nullptr;
}

size_t
Shot::ComputeNumValidPoints() const
{
  return points_.size() - std::count(points_.cbegin(), points_.cend(), nullptr);
}


void
Shot::InitAndTakeDatastructures(std::vector<cv::KeyPoint> keypts, cv::Mat descriptors)
{
  assert(keypoints.size() == descriptors.rows);

  std::swap(keypts, keypoints_);
  std::swap(descriptors, descriptors_);
  num_keypts_ = keypoints_.size();
  points_.resize(num_keypts_, nullptr);
}

} //namespace reconstruction

