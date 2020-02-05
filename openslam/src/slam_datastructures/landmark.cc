#include "landmark.h"
#include "keyframe.h"
#include <iostream>
namespace cslam
{
void 
Landmark::add_observation(KeyFrame* keyfrm, size_t idx) {
    // std::lock_guard<std::mutex> lock(mtx_observations_);
    std::cout << "Trying to add: " << keyfrm->kf_id_ << "/" << idx << std::endl;
    if (observations_.count(keyfrm)) {
        return;
    }
    observations_[keyfrm] = idx;
    std::cout << "Added: " << keyfrm->kf_id_ << "/" << idx << std::endl;
    // if (0 <= keyfrm->stereo_x_right_.at(idx)) {
    //     num_observations_ += 2;
    // }
    // else {
        num_observations_ += 1;
    // }
    // const auto class_id = keyfrm->keypts_[idx].class_id;
    // if (class_id > 0 && class_id < 255) num_map_feature_obs_2_++;
}
}