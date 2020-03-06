#include <map/shot.h>
#include <map/camera.h>
#include <map/landmark.h>
#include <algorithm>
namespace map
{
Shot::Shot(const ShotId shot_id, const ShotCamera& camera, const Pose& pose, const std::string& name):
            id_(shot_id), shot_name_(name), camera_(camera), pose_(pose)
{
  
}

void
Shot::RemoveLandmarkObservation(const FeatureId id)
{
  landmarks_.at(id) = nullptr;
}

size_t
Shot::ComputeNumValidLandmarks() const
{
  return landmarks_.size() - std::count(landmarks_.cbegin(), landmarks_.cend(), nullptr);
}


void
Shot::InitAndTakeDatastructures(std::vector<cv::KeyPoint> keypts, cv::Mat descriptors)
{
  assert(keypoints.size() == descriptors.rows);

  std::swap(keypts, keypoints_);
  std::swap(descriptors, descriptors_);
  num_keypts_ = keypoints_.size();
  landmarks_.resize(num_keypts_, nullptr);
}

} //namespace map

