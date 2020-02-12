#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include "types.h"

namespace py = pybind11;

namespace openvslam{ namespace feature { class orb_extractor; }}

namespace cslam
{
class KeyFrame;
class Landmark;
// class BrownPerspectiveCamera;
class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Frame(const csfm::pyarray_uint8 image, const csfm::pyarray_uint8 mask, const std::string& img_name,
          const size_t frame_id, openvslam::feature::orb_extractor* orb_extractor);
    void add_landmark(Landmark* lm, size_t idx);
    py::object get_lm_and_obs(); //cslam::Frame& frame) const;
    const std::string im_name;
    const size_t frame_id;
    KeyFrame* mParentKf;
    std::vector<cv::KeyPoint> keypts_; // extracted keypoints
    std::vector<cv::KeyPoint> undist_keypts_; // undistorted keypoints
    cv::Mat descriptors_;
    std::vector<Landmark*> landmarks_;
    std::vector<bool> outlier_flags_;
    std::vector<std::vector<std::vector<unsigned int>>> keypts_indices_in_cells_;
    //! bearing vectors
    // Eigen::eigen_alloc_vector<Eigen::Vector3f> bearings_;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings_;
    const std::vector<float> scale_factors_;
    py::object getKptsAndDescPy() const;
    py::object getKptsUndist() const;
    py::object getKptsPy() const;
    py::object getDescPy() const;
    py::object get_valid_keypts() const;
    std::vector<size_t> get_valid_idx() const;
    void set_outlier(const std::vector<size_t>& invalid_ids);
    std::vector<Landmark*> get_valid_lms();
    size_t num_keypts_;
    Eigen::Vector3f get_cam_center() const { return cam_center_; }
    Eigen::Matrix4f getPose() const { return T_wc; }
    Eigen::Matrix4f getTcw() const { return T_cw; }
    // sets cam to world
    // void setPose(const Eigen::Matrix4f& pose)
    // {
    //     T_wc = pose;
    //     cam_pose_cw_ = T_wc.inverse();
    //     // cam_center_ = T_cw.block<3,1>(0,3);//-rot_wc * trans_cw;
    // }
    void setPose(const Eigen::Matrix4f& pose)
    {
        T_wc = pose;
        T_cw = T_wc.inverse();
        // cam_pose_cw_ = T_wc.inverse();
        cam_center_ = T_cw.block<3,1>(0,3);//-rot_wc * trans_cw;
    }
    size_t discard_outliers();
    // ORB scale pyramid information
    //! number of scale levels
    size_t num_scale_levels_;
    //! scale factor
    float scale_factor_;
    //! log scale factor
    float log_scale_factor_;
    void update_orb_info(openvslam::feature::orb_extractor* orb_extractor);
private:
    Eigen::Matrix4f T_cw, T_wc;
    Eigen::Vector3f cam_center_;

};
}