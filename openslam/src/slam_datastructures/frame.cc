#include "frame.h"
#include "types.h"
namespace cslam
{
Frame::Frame(const std::string& img_name, const size_t frame_id):
    mImgName(img_name), mFrameId(frame_id)
{}

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
    cv::Mat keys(mKeyPts.size(), 5, CV_32F);
    for (int i = 0; i < (int) undist_keypts_.size(); ++i) {
        keys.at<float>(i, 0) = mKeyPts[i].pt.x;
        keys.at<float>(i, 1) = mKeyPts[i].pt.y;
        keys.at<float>(i, 2) = mKeyPts[i].size;
        keys.at<float>(i, 3) = mKeyPts[i].angle;
        keys.at<float>(i, 4) = mKeyPts[i].octave;
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}
py::object
Frame::getKptsAndDescPy() const
{
    // Convert to numpy.
    cv::Mat keys(mKeyPts.size(), 5, CV_32F);
    for (int i = 0; i < (int) mKeyPts.size(); ++i) {
        keys.at<float>(i, 0) = mKeyPts[i].pt.x;
        keys.at<float>(i, 1) = mKeyPts[i].pt.y;
        keys.at<float>(i, 2) = mKeyPts[i].size;
        keys.at<float>(i, 3) = mKeyPts[i].angle;
        keys.at<float>(i, 4) = mKeyPts[i].octave;
    }

    py::list retn;
    retn.append(csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols));
    retn.append(csfm::py_array_from_data(descriptors_.ptr<unsigned char>(0), descriptors_.rows, descriptors_.cols));
    return retn;
}
          
}