#include <slam/slam_utilities.h>
#include <map/landmark.h>
#include <Eigen/Core>
namespace slam
{
class PySlamUtilities
{
public:
static Eigen::MatrixXf GetUndistortedKeyptsFromShot(const map::Shot& shot)
{
    return SlamUtilities::ConvertOpenCVKptsToEigen(shot.slam_data_.undist_keypts_);
}

static Eigen::MatrixXf GetKeyptsFromShot(const map::Shot& shot)
{
    return SlamUtilities::ConvertOpenCVKptsToEigen(shot.GetKeyPoints());
}

static void SetDescriptorFromObservations(map::Landmark& landmark)
{
    const auto& observations = landmark.GetObservations();
    if (observations.empty()) {
        return;
    }
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(observations.size());
    for (const auto& observation : observations) {
        auto shot = observation.first;
        const auto idx = observation.second;
        descriptors.push_back(shot->GetDescriptor(idx));
    }
    const auto median_idx = GuidedMatcher::ComputeMedianDescriptorIdx(descriptors);
    landmark.slam_data_.descriptor_ = descriptors[median_idx].clone();
}


static void SetNormalAndDepthFromObservations(map::Landmark& landmark, const std::vector<float>& scale_factors)
{
    const auto& observations = landmark.GetObservations();
    if (observations.empty()) {
        return;
    }
    Eigen::Vector3d mean_normal = Eigen::Vector3d::Zero();
    unsigned int num_observations = 0;
    for (const auto& observation : observations) {
        const auto shot = observation.first;
        const Eigen::Vector3d cam_center = shot->GetPose().GetOrigin(); //.cast<float>();//get_cam_center();
        const Eigen::Vector3d normal = landmark.GetGlobalPos() - cam_center;
        mean_normal = mean_normal + normal / normal.norm();
        ++num_observations;
    }
    const auto dist = landmark.ComputeDistanceFromRefFrame();
    auto* ref_shot = landmark.GetRefShot();
    const auto ref_obs_idx = observations.at(ref_shot);
    const auto scale_level = ref_shot->slam_data_.undist_keypts_.at(ref_obs_idx).octave;
    // const auto scale_level = ref_shot_->undist_keypts_.at(observations_.at(ref_shot_)).octave;
    const auto scale_factor = scale_factors.at(scale_level);
    const auto num_scale_levels = scale_factors.size();

    landmark.slam_data_.max_valid_dist_ = dist * scale_factor;
    landmark.slam_data_.min_valid_dist_ = landmark.slam_data_.max_valid_dist_ / scale_factors.at(num_scale_levels - 1);
    landmark.slam_data_.mean_normal_ = mean_normal / num_observations;
}

};
}