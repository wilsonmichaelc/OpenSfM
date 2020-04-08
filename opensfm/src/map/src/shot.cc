#include <map/shot.h>
#include <map/camera.h>
#include <map/landmark.h>
#include <algorithm>
namespace map
{
Shot::Shot(const ShotId shot_id, const ShotCamera& camera, const Pose& pose, const std::string& name):
            id_(shot_id), name_(name), camera_(camera), pose_(pose)
{
  
}

size_t
Shot::ComputeNumValidLandmarks() const
{
  return landmarks_.size() - std::count(landmarks_.cbegin(), landmarks_.cend(), nullptr);
}

void
Shot::InitKeyptsAndDescriptors(const size_t n_keypts)
{
  if (n_keypts > 0)
  {
    num_keypts_ = n_keypts;
    landmarks_.resize(num_keypts_, nullptr);
    keypoints_.resize(num_keypts_);
    descriptors_ = cv::Mat(n_keypts, 32, CV_8UC1, cv::Scalar(0));
  }
}

void
Shot::InitAndTakeDatastructures(std::vector<cv::KeyPoint> keypts, cv::Mat descriptors)
{
  assert(keypts.size() == descriptors.rows);

  std::swap(keypts, keypoints_);
  std::swap(descriptors, descriptors_);
  num_keypts_ = keypoints_.size();
  landmarks_.resize(num_keypts_, nullptr);
}

} //namespace map

