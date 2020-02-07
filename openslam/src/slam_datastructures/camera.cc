#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "camera.h"
#include "frame.h"
#include "third_party/openvslam/util/guided_matching.h"
namespace cslam
{


void
BrownPerspectiveCamera::undistKeyptsFrame(Frame& frame) const
{
    undistKeypts(frame.keypts_, frame.undist_keypts_);
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
    std::cout << "undist_keypts: " << undist_keypts.size() << std::endl;
    undist_keypts.resize(num_keypts);
    std::cout << "num_keypts: " << num_keypts << " keypts: " << keypts.size() << std::endl;
    for(int idx = 0; idx < num_keypts; idx++)
    {
        undist_keypts.at(idx).pt.x = upTmp.at<float>(idx, 0);
        undist_keypts.at(idx).pt.y = upTmp.at<float>(idx, 1);
        undist_keypts.at(idx).angle = keypts.at(idx).angle;
        undist_keypts.at(idx).size = keypts.at(idx).size;
        undist_keypts.at(idx).octave = keypts.at(idx).octave;
    }
    std::cout << "undist_keypts: " << undist_keypts.size() << std::endl;
}

bool 
BrownPerspectiveCamera::reproject_to_image(const Eigen::Matrix3f& R_cw, const Eigen::Vector3f& t_cw, const Eigen::Vector3f& ptWorld,
                                            const cslam::GridParameters& gridParams,
                                            Eigen::Vector2f& pt2D) const
{
    //first, transform pt3D into cam
    const Eigen::Vector3f ptCam = R_cw*ptWorld + t_cw;
    std::cout << "ptCam: " << ptCam << " R: " << R_cw << " t_cw: " << t_cw << std::endl;
    std::cout << "K: " << K <<  "K_pixel: " << K_pixel << std::endl;
    //check z coordinate
    if (ptCam[2] < 0.0) return false;
    

    // //now reproject to image
    // const float z_inv = 1.0/ptCam[2];
    // pt2D[0] = fx*ptCam[0]*z_inv + cx;
    // pt2D[1] = fy*ptCam[1]*z_inv + cy;
    pt2D = (K_pixel_eig*ptCam).hnormalized();
    std::cout << "pt2D: " << pt2D << " f: " << fx << ", " << fy << std::endl;

    //check boundaries
    return gridParams.in_grid(pt2D);
    // return gridParams.img_min_width_ < pt2D[0] && gridParams.img_max_width_ > pt2D[0] &&
    //        gridParams.img_min_height_ < pt2D[1] && gridParams.img_max_height_ > pt2D[1]; 
}

}