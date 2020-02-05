#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace cslam
{
class KeyFrame;
class Landmark;
class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Frame(const std::string& img_name, const size_t frame_id);
    // std::string getImgName() const { return mImgName; };
    // size_t getFrameId() const { return mFrameId; };
    const std::string mImgName;
    const size_t mFrameId;
    KeyFrame* mParentKf;
    std::vector<cv::KeyPoint> mKeyPts; // extracted keypoints
    std::vector<cv::KeyPoint> undist_keypts_; // undistorted keypoints
    cv::Mat descriptors_;
    std::vector<Landmark*> landmarks_;
    std::vector<std::vector<std::vector<unsigned int>>> keypts_indices_in_cells_;
    //! bearing vectors
    // Eigen::eigen_alloc_vector<Eigen::Vector3f> bearings_;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings_;
    size_t mNumKeypts;

    py::object getKptsAndDescPy() const;
    py::object getKptsUndist() const;
    py::object getKptsPy() const;
    py::object getDescPy() const;
};
}