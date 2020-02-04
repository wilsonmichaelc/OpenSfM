#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "camera.h"
#include "frame.h"
namespace cslam
{


void
BrownPerspectiveCamera::undistKeyptsFrame(Frame& frame) const
{
    undistKeypts(frame.mKeyPts, frame.mKeyPtsUndist);
}

void 
BrownPerspectiveCamera::undistKeypts(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::KeyPoint>& undist_keypts) const
{
    //TODO: 0 distortion?

    const auto num_keypts = keypts.size();
    // Fill matrix with points
    cv::Mat upTmp(num_keypts,2,CV_32F);
    for(int i=0; i<num_keypts; i++)
    {
        upTmp.at<float>(i,0)=keypts[i].pt.x;
        upTmp.at<float>(i,1)=keypts[i].pt.y;
    }
        // Undistort points
    upTmp=upTmp.reshape(2);
    cv::undistortPoints(upTmp,upTmp,K,distCoeff,cv::Mat(),K);
    upTmp=upTmp.reshape(1);

    undist_keypts.resize(num_keypts);
    for(int idx = 0; idx < num_keypts; idx++)
    {
        undist_keypts.at(idx).pt.x = upTmp.at<float>(idx, 0);
        undist_keypts.at(idx).pt.y = upTmp.at<float>(idx, 1);
        undist_keypts.at(idx).angle = keypts.at(idx).angle;
        undist_keypts.at(idx).size = keypts.at(idx).size;
        undist_keypts.at(idx).octave = keypts.at(idx).octave;
    }
}

}