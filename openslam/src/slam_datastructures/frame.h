#pragma once
// #include <string>
#include <vector>
// #include <opencv2/opencv.hpp>
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
    Frame(const std::string& img_name, const size_t frame_id);
    // std::string getImgName() const { return mImgName; };
    // size_t getFrameId() const { return mFrameId; };
    const std::string mImgName;
    const size_t mFrameId;
    KeyFrame* mParentKf;
    std::vector<cv::KeyPoint> mKeyPts; // extracted keypoints
    std::vector<cv::KeyPoint> mKeyPtsUndist; // undistorted keypoints
    cv::Mat mDescriptors;
    std::vector<Landmark*> mLandmarks;
    std::vector<std::vector<std::vector<unsigned int>>> mKeyptIndicesInCells; 
    size_t mNumKeypts;

    py::object getKptsAndDescPy() const;
    py::object getKptsUndist() const;
    py::object getKptsPy() const;
    py::object getDescPy() const;
};
}