#pragma once
#include <vector>
#include <Eigen/Eigen>
#include <opencv2/core.hpp>
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
    Eigen::Vector3f get_cam_center() const { return cam_center_; }
    const std::vector<float> scale_factors_;
    const size_t num_scale_levels_;
    void add_landmark(Landmark* lm, const size_t idx);
private:
    Eigen::Matrix4f T_wc; // camera to world transformation, pose
    Eigen::Matrix4f T_cw; // world to camera transformation
    Eigen::Vector3f cam_center_;

};
}