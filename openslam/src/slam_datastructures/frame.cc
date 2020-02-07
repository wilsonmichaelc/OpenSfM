#include "frame.h"
#include "landmark.h"
#include "types.h"
#include <pybind11/pybind11.h>
#include "third_party/openvslam/feature/orb_extractor.h"

namespace cslam
{
Frame::Frame(const csfm::pyarray_uint8 image, const csfm::pyarray_uint8 mask,
             const std::string& img_name, const size_t frame_id, 
             openvslam::feature::orb_extractor* orb_extractor):
    mImgName(img_name), mFrameId(frame_id), scale_factors_(orb_extractor->get_scale_factors()),
    T_wc(Eigen::Matrix4f::Identity()), cam_pose_cw_(Eigen::Matrix4f::Identity())
{
    orb_extractor->extract_orb_py2(image, mask, *this);
    num_keypts_ = keypts_.size();
    std::cout << "Allocating " << num_keypts_ << " for frame: " << img_name << std::endl;
    landmarks_ = std::vector<Landmark*>(num_keypts_, nullptr);
    outlier_flags_ = std::vector<bool>(num_keypts_, false);
}
void 
Frame::add_landmark(Landmark* lm, size_t idx)
{
    std::cout << "idx: " << idx << ", " << num_keypts_ << std::endl;
    // landmarks_[idx] = lm;
    // outlier_flags_[idx] = false;
    landmarks_.at(idx) = lm;
    outlier_flags_.at(idx) = false;
}

py::object
Frame::getKptsUndist() const
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
Frame::getDescPy() const
{
    return csfm::py_array_from_data(descriptors_.ptr<unsigned char>(0), descriptors_.rows, descriptors_.cols);
}
py::object
Frame::getKptsPy() const
{
    // Convert to numpy.
    cv::Mat keys(keypts_.size(), 5, CV_32F);
    for (int i = 0; i < (int) undist_keypts_.size(); ++i) {
        keys.at<float>(i, 0) = keypts_[i].pt.x;
        keys.at<float>(i, 1) = keypts_[i].pt.y;
        keys.at<float>(i, 2) = keypts_[i].size;
        keys.at<float>(i, 3) = keypts_[i].angle;
        keys.at<float>(i, 4) = keypts_[i].octave;
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}
py::object
Frame::getKptsAndDescPy() const
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

    py::list retn;
    retn.append(csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols));
    retn.append(csfm::py_array_from_data(descriptors_.ptr<unsigned char>(0), descriptors_.rows, descriptors_.cols));
    return retn;
}
          
}