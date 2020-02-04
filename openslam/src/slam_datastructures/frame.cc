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
    cv::Mat keys(mKeyPtsUndist.size(), 5, CV_32F);
    for (int i = 0; i < (int) mKeyPtsUndist.size(); ++i) {
        keys.at<float>(i, 0) = mKeyPtsUndist[i].pt.x;
        keys.at<float>(i, 1) = mKeyPtsUndist[i].pt.y;
        keys.at<float>(i, 2) = mKeyPtsUndist[i].size;
        keys.at<float>(i, 3) = mKeyPtsUndist[i].angle;
        keys.at<float>(i, 4) = mKeyPtsUndist[i].octave;
    }
    return csfm::py_array_from_data(keys.ptr<float>(0), keys.rows, keys.cols);
}
py::object
Frame::getDescPy() const
{
    return csfm::py_array_from_data(mDescriptors.ptr<unsigned char>(0), mDescriptors.rows, mDescriptors.cols);
}
py::object
Frame::getKptsPy() const
{
    // Convert to numpy.
    cv::Mat keys(mKeyPts.size(), 5, CV_32F);
    for (int i = 0; i < (int) mKeyPtsUndist.size(); ++i) {
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
    retn.append(csfm::py_array_from_data(mDescriptors.ptr<unsigned char>(0), mDescriptors.rows, mDescriptors.cols));
    return retn;
}
          
}