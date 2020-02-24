#include "keyframe.h"
#include "frame.h"
#include "landmark.h"
namespace cslam
{

KeyFrame::KeyFrame(const size_t kf_id, const Frame& frame):
    kf_id_(kf_id), src_frm_id_(frame.frame_id),
    im_name_(frame.im_name),
    keypts_(frame.keypts_), undist_keypts_(frame.undist_keypts_),
    bearings_(frame.bearings_), descriptors_(frame.descriptors_),
    landmarks_(frame.landmarks_), 
    // ORB scale pyramid
    num_scale_levels_(frame.num_scale_levels_), scale_factor_(frame.scale_factor_),
    log_scale_factor_(frame.log_scale_factor_), scale_factors_(frame.scale_factors_),
    level_sigma_sq_(frame.level_sigma_sq_), inv_level_sigma_sq_(frame.inv_level_sigma_sq_),
    keypts_indices_in_cells_(frame.keypts_indices_in_cells_), num_keypts_(landmarks_.size())
{

}

std::vector<size_t>
KeyFrame::compute_local_keyframes() const
{
    std::vector<bool> seen(kf_id_+1, false);
    for (auto lm : landmarks_)
    {
        if (lm != nullptr)
        {
            // const auto& m = lm->get_observations();
            for (const auto& elem : lm->get_observations())
            {
                seen.at(elem.first->kf_id_) = true;
            }

        }
    }
    //TODO: be careful when we remove KFs!!!!
    std::vector<size_t> seen_idx;
    seen_idx.reserve(kf_id_+1);
    for (size_t idx = 0; idx < seen.size(); ++idx)
    {
        if (seen.at(idx)) seen_idx.push_back(idx);
    }
    return seen_idx;
}

py::object
KeyFrame::getKptsUndist() const
{
    // Convert to numpy.
    cv::Mat keys(undist_keypts_.size(), 5, CV_32F);
    for (int i = 0; i < (int) undist_keypts_.size(); ++i) {
        keys.at<float>(i, 0) = undist_keypts_[i].pt.x;
        keys.at<float>(i, 1) = undist_keypts_[i].pt.y;
        keys.at<float>(i, 2) = undist_keypts_[i].size;
        keys.at<float>(i, 3) = undist_keypts_[i].angle;
        keys.at<float>(i, 4) = undist_keypts_[i].octave;
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}

py::object
KeyFrame::getDescPy() const
{
    return csfm::py_array_from_data(descriptors_.ptr<unsigned char>(0), descriptors_.rows, descriptors_.cols);
}

py::object
KeyFrame::getKptsPy() const
{
    // Convert to numpy.
    cv::Mat keys(keypts_.size(), 5, CV_32F);
    for (int i = 0; i < (int) keypts_.size(); ++i) {
        keys.at<float>(i, 0) = keypts_[i].pt.x;
        keys.at<float>(i, 1) = keypts_[i].pt.y;
        keys.at<float>(i, 2) = keypts_[i].size;
        keys.at<float>(i, 3) = keypts_[i].angle;
        keys.at<float>(i, 4) = keypts_[i].octave;
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}
void
KeyFrame::add_landmark(Landmark* lm, const size_t idx)
{
    landmarks_[idx] = lm;
    // std::cout << "landmarks_["<< idx << "]: " <<  landmarks_[idx] << " and " << lm << " kf: " << kf_id_ << "ptr: " << this << std::endl;
}

size_t 
KeyFrame::get_num_tracked_lms(const size_t min_num_obs_thr) const
{
    size_t num_tracked_lms = 0;
    if (0 < min_num_obs_thr) {
        for (const auto lm : landmarks_) {
            if (lm == nullptr) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            if (min_num_obs_thr <= lm->num_observations()) {
                ++num_tracked_lms;
            }
        }
    }
    else {
        for (const auto lm : landmarks_) {
            if (lm == nullptr) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            ++num_tracked_lms;
        }
    }

    return num_tracked_lms;

}

float 
KeyFrame::compute_median_depth(const bool abs) const
{
    std::vector<float> depths;
    depths.reserve(landmarks_.size());
    const Eigen::Vector3f rot_cw_z_row = T_cw.block<1, 3>(2, 0);
    // const Eigen::Vector3f rot_cw_z_row = cam_pose_cw_.block<1, 3>(2, 0);

    // T_cw
    const float trans_cw_z = T_cw(2, 3);
    for (const auto lm : landmarks_)
    {
        if (lm == nullptr) continue;
        const float pos_c_z = rot_cw_z_row.dot(lm->get_pos_in_world())+trans_cw_z;
        depths.push_back(abs ? std::abs(pos_c_z) : pos_c_z);
    }
    std::sort(depths.begin(), depths.end());
    return depths.at((depths.size() - 1) / 2);
}

std::vector<Landmark*>
KeyFrame::get_valid_lms()
{
    std::vector<Landmark*> landmark;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> obs; //x, y, scale
    for (size_t i = 0; i < num_keypts_; ++i)//(auto lm : landmarks_)
    {
        auto lm = landmarks_[i];
        if (lm == nullptr) continue;
        landmark.push_back(lm);
        const auto kp = keypts_[i];
        obs.push_back(Eigen::Vector3f(kp.pt.x, kp.pt.y, kp.size));
    }
    return landmark;
}
py::object
KeyFrame::get_valid_keypts() const
{
    const auto n_valid_pts = num_keypts_ - std::count(landmarks_.cbegin(), landmarks_.cend(),nullptr);
    // Convert to numpy.
    cv::Mat keys(n_valid_pts, 3, CV_32F);
    size_t idx2{0};
    for (size_t i = 0; i < keypts_.size(); ++i) {
    if (landmarks_[i] != nullptr)
        {
            keys.at<float>(idx2, 0) = keypts_[i].pt.x;
            keys.at<float>(idx2, 1) = keypts_[i].pt.y;
            keys.at<float>(idx2, 2) = keypts_[i].size;
            idx2++;
        }
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}

std::vector<size_t> 
KeyFrame::get_valid_idx() const
{
    std::vector<size_t> valid_idx;
    for (size_t i = 0; i < num_keypts_; ++i)
    {
        auto lm = landmarks_[i];
        if (lm == nullptr) continue;
        valid_idx.push_back(i);
    }
    return valid_idx;
}


// std::vector<Landmark*> 
// KeyFrame::update_lms_after_kf_insert()
// {
//     std::vector<Landmark*> landmarks_to_clean;
//     // landmarks_
//     for (unsigned int idx = 0; idx < landmarks_.size(); ++idx) {
//         auto lm = landmarks_.at(idx);
//         if (!lm) {
//             continue;
//         }
//         if (lm->will_be_erased()) {
//             continue;
//         }

//         // if `lm` does not have the observation information from `cur_keyfrm_`,
//         // add the association between the keyframe and the landmark
//         if (lm->is_observed_in_keyframe(this)) {
//             // if `lm` is correctly observed, make it be checked by the local map cleaner
//             // local_map_cleaner_->add_fresh_landmark(lm);
//             landmarks_to_clean.push_back(lm);
//             continue;
//         }

//         // update connection
//         lm->add_observation(this, idx);
//         // update geometry
//         lm->update_normal_and_depth();
//         lm->compute_descriptor();
//     }
//     return landmarks_to_clean;
// }

}