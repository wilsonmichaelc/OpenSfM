#include <slam/slam_utilities.h>
#include <map/landmark.h>
#include <bundle/bundle_adjuster.h>
#include <foundation/types.h>
#include <Eigen/Core>
namespace slam
{
class PySlamUtilities
{
public:
  static Eigen::MatrixXf GetUndistortedKeyptsFromShot(const map::Shot &shot)
  {
    return SlamUtilities::ConvertOpenCVKptsToEigen(shot.slam_data_.undist_keypts_);
  }

  static Eigen::MatrixXf GetKeyptsFromShot(const map::Shot &shot)
  {
    return SlamUtilities::ConvertOpenCVKptsToEigen(shot.GetKeyPoints());
  }

  static void SetDescriptorFromObservations(map::Landmark &landmark)
  {
    SlamUtilities::SetDescriptorFromObservations(landmark);
    // const auto& observations = landmark.GetObservations();
    // if (observations.empty()) {
    //     return;
    // }
    // std::vector<cv::Mat> descriptors;
    // descriptors.reserve(observations.size());
    // for (const auto& observation : observations) {
    //     auto shot = observation.first;
    //     const auto idx = observation.second;
    //     descriptors.push_back(shot->GetDescriptor(idx));
    // }
    // const auto median_idx = GuidedMatcher::ComputeMedianDescriptorIdx(descriptors);
    // landmark.slam_data_.descriptor_ = descriptors[median_idx].clone();
  }

  static void SetNormalAndDepthFromObservations(map::Landmark &landmark, const std::vector<float> &scale_factors)
  {
    SlamUtilities::SetNormalAndDepthFromObservations(landmark, scale_factors);
  }

  static py::object GetValidKeypts(const map::Shot &shot)
  {
    const auto &landmarks = shot.GetLandmarks();
    const auto n_landmarks = landmarks.size();
    // const auto n_valid_pts = n_landmarks - std::count(landmarks.cbegin(), landmarks.cend(),nullptr);
    const auto &keypts = shot.GetKeyPoints();
    // Convert to numpy.
    cv::Mat keys(n_landmarks, 3, CV_32F);
    size_t idx2{0};
    for (size_t i = 0; i < n_landmarks; ++i)
    {
      if (landmarks[i] != nullptr)
      {
        keys.at<float>(idx2, 0) = keypts[i].pt.x;
        keys.at<float>(idx2, 1) = keypts[i].pt.y;
        keys.at<float>(idx2, 2) = keypts[i].size;
        idx2++;
      }
    }
    return foundation::py_array_from_data(keys.ptr<float>(0), idx2, keys.cols);
  }

  static auto
  UpdateLocalKeyframes(const map::Shot &shot)
  {
    return SlamUtilities::update_local_keyframes(shot);
  }

  static auto
  UpdateLocalLandmarks(const std::vector<map::Shot *> &local_keyframes)
  {
    return SlamUtilities::update_local_landmarks(local_keyframes);
  }

  static auto
  MatchShotToLocalLandmarks(map::Shot &shot, const GuidedMatchingWrapper &matcher)
  {
    return SlamUtilities::MatchShotToLocalMap(shot, matcher.matcher_);
  }

  static auto ComputeMinMaxDepthInShot(const map::Shot &shot)
  {
    return SlamUtilities::ComputeMinMaxDepthInShot(shot);
  }

  static Eigen::Matrix3d create_E_21(const Eigen::Matrix3d &rot_1w, const Eigen::Vector3d &trans_1w,
                                     const Eigen::Matrix3d &rot_2w, const Eigen::Vector3d &trans_2w)
  {
    return SlamUtilities::create_E_21(rot_1w, trans_1w, rot_2w, trans_2w);
  }
};
} // namespace slam