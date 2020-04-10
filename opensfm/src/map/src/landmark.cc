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
// void
// Landmark::ComputeDescriptor()
// {
//     if (observations_.empty()) return;
//     // std::cout << "compute_descriptor: " << lm_id_ << std::endl;

//     // 対応している特徴点の特徴量を集める
//     std::vector<cv::Mat> descriptors;
//     descriptors.reserve(observations_.size());
//     for (const auto& observation : observations_) {
//         auto keyfrm = observation.first;
//         const auto idx = observation.second;
//         // if (!keyfrm->will_be_erased()) {
//             descriptors.push_back(keyfrm->GetDescriptor(idx));
//             // std::cout << "keyfrm->descriptors_.row(" << idx << "): " << keyfrm->descriptors_.row(idx) << std::endl;
//         // }
//     }

//     // ハミング距離の中央値を計算
//     // Calculate median Hamming distance

//     // まず特徴量間の距離を全組み合わせで計算
//     // First, calculate the distance between features in all combinations
//     const auto num_descs = descriptors.size();
//     // std::cout << "Computing: " << num_descs << std::endl;
//     std::vector<std::vector<unsigned int>> hamm_dists(num_descs, std::vector<unsigned int>(num_descs));
//     for (unsigned int i = 0; i < num_descs; ++i) {
//         hamm_dists.at(i).at(i) = 0;
//         for (unsigned int j = i + 1; j < num_descs; ++j) {
//             const auto dist = slam::GuidedMatcher::compute_descriptor_distance_32(descriptors.at(i), descriptors.at(j));
//             hamm_dists.at(i).at(j) = dist;
//             hamm_dists.at(j).at(i) = dist;
//             // std::cout << "i/j: " << i << "/" << j << " dist: " << dist << std::endl;
//         }
//     }

//     // 中央値に最も近いものを求める
//     // Find the closest to the median
//     unsigned int best_median_dist = slam::GuidedMatcher::MAX_HAMMING_DIST;
//     unsigned int best_idx = 0;
//     for (unsigned idx = 0; idx < num_descs; ++idx) {
//         std::vector<unsigned int> partial_hamm_dists(hamm_dists.at(idx).begin(), hamm_dists.at(idx).begin() + num_descs);
//         std::sort(partial_hamm_dists.begin(), partial_hamm_dists.end());
//         const auto median_dist = partial_hamm_dists.at(static_cast<unsigned int>(0.5 * (num_descs - 1)));
//         // std::cout << "median_dist: " << median_dist << "num_descs: " << num_descs << std::endl;
//         if (median_dist < best_median_dist) {
//             best_median_dist = median_dist;
//             best_idx = idx;
//         }
//     }
//     // std::cout << "Final descriptor at " << best_idx << " desc: " << descriptors.at(best_idx);
//     slam_data_.descriptor_ = descriptors.at(best_idx).clone();
// }
// void 
// Landmark::UpdateNormalAndDepth()
// {
//     if (observations_.empty()) {
//         return;
//     }

//     Eigen::Vector3d mean_normal = Eigen::Vector3f::Zero();
//     unsigned int num_observations = 0;
//     for (const auto& observation : observations_) {
//         const auto shot = observation.first;
//         const Eigen::Vector3d cam_center = shot->GetPose().GetOrigin(); //.cast<float>();//get_cam_center();
//         const Eigen::Vector3d normal = global_pos_ - cam_center;
//         mean_normal = mean_normal + normal / normal.norm();
//         ++num_observations;
//     }
//     //TODO: num_obs == observations.size()
//     const Eigen::Vector3d cam_to_lm_vec = global_pos_ - ref_shot_->GetPose().GetOrigin(); //.cast<float>();
//     const auto dist = cam_to_lm_vec.norm();
//     const auto scale_level = ref_shot_->undist_keypts_.at(observations_.at(ref_shot_)).octave;
//     const auto scale_factor = ref_shot_->scale_factors_.at(scale_level);
//     const auto num_scale_levels = ref_shot_->num_scale_levels_;

//     slam_data_.max_valid_dist_ = dist * scale_factor;
//     slam_data_.min_valid_dist_ = slam_data_.max_valid_dist_ / ref_keyfrm_->scale_factors_.at(num_scale_levels - 1);
//     slam_data_.mean_normal_ = mean_normal / num_observations;
// }

}; //namespace map
