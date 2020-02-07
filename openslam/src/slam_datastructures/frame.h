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
class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Frame(const csfm::pyarray_uint8 image, const csfm::pyarray_uint8 mask, const std::string& img_name,
          const size_t frame_id, openvslam::feature::orb_extractor* orb_extractor);
    void add_landmark(Landmark* lm, size_t idx);
    const std::string mImgName;
    const size_t mFrameId;
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
    size_t num_keypts_;

    Eigen::Matrix4f getPose() const { return T_wc; }
    void setPose(const Eigen::Matrix4f& pose)
    {
        T_wc = pose;
        cam_pose_cw_ = T_wc.inverse();
        // cam_center_ = T_cw.block<3,1>(0,3);//-rot_wc * trans_cw;
    }
private:
    Eigen::Matrix4f cam_pose_cw_, T_wc;
};
}