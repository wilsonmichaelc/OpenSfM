#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include "types.h"
namespace py = pybind11;
namespace cslam
{
class Frame;
class Landmark;
class KeyFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    KeyFrame() = delete;
    KeyFrame(const size_t kf_id, const Frame& frame);
    size_t kf_id_;
    const size_t src_frm_id_;

    //! keypoints of monocular or stereo left image
    const std::vector<cv::KeyPoint> keypts_;
    //! undistorted keypoints of monocular or stereo left image
    const std::vector<cv::KeyPoint> undist_keypts_;
    //! bearing vectors
    const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings_;
    
    //! keypoint indices in each of the cells
    const std::vector<std::vector<std::vector<unsigned int>>> keypt_indices_in_cells_;
    //! descriptors
    const cv::Mat descriptors_;
    std::vector<Landmark*> landmarks_;

    Eigen::Matrix4f getPose() const { return T_wc; }
    void setPose(const Eigen::Matrix4f& pose)
    {
        T_wc = pose;
        T_cw = T_wc.inverse();
        cam_center_ = T_cw.block<3,1>(0,3);//-rot_wc * trans_cw;
    }
    Eigen::Matrix4f getTcw() const { return T_cw; }
    Eigen::Vector3f get_cam_center() const { return cam_center_; }
    const std::vector<float> scale_factors_;
    // ORB scale pyramid information
    //! number of scale levels
    size_t num_scale_levels_;
    //! scale factor
    float scale_factor_;
    //! log scale factor
    float log_scale_factor_;
    void add_landmark(Landmark* lm, const size_t idx);
    size_t get_num_tracked_lms(const size_t min_num_obs_thr) const;
    // basically store_new_keyframe
    // std::vector<Landmark*> update_lms_after_kf_insert();
    void erase_landmark_with_index(const unsigned int idx) 
    {
        // std::lock_guard<std::mutex> lock(mtx_observations_);
        landmarks_.at(idx) = nullptr;
    }
    float compute_median_depth(const bool abs) const;
    py::object getKptsUndist() const;
    py::object getKptsPy() const;
    py::object getDescPy() const;
private:
    Eigen::Matrix4f T_wc; // camera to world transformation, pose
    Eigen::Matrix4f T_cw; // world to camera transformation
    Eigen::Vector3f cam_center_;

};
}