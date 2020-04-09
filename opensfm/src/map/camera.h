#pragma once
#include <map/defines.h>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
namespace map
{
class Camera
{
public:
    Camera(const size_t width_, const size_t height_, const std::string& projection_type_):
           width(width_), height(height_), projectionType(projection_type_)
    {}

    const size_t width;
    const size_t height;
    const std::string projectionType;
  virtual void UndistortedKeyptsToBearings(const std::vector<cv::KeyPoint>& undistKeypts,
                                      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings) const;
  virtual void UndistortKeypts(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::KeyPoint>& undist_keypts) const;

};

class BrownPerspectiveCamera : public Camera
{
public:
  BrownPerspectiveCamera(const size_t width_, const size_t height_, const std::string& projection_type_,
                         const float fx_, const float fy_, const float cx_, const float cy_,
                         const float k1_, const float k2_, const float p1_, const float p2_, const float k3_);
  virtual void UndistortedKeyptsToBearings(const std::vector<cv::KeyPoint>& undistKeypts,
                                      std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings) const;
  virtual void UndistortKeypts(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::KeyPoint>& undist_keypts) const;

  float fx_p, fy_p; // focal lengths in pixels
  float cx_p, cy_p; // principal points in pixels
  
  const float fx, fy; // focal lengths
  const float cx, cy; // principal points
  const float k1, k2, p1, p2, k3; // distortion coefficients
  cv::Mat K, K_pixel; //intrinsic camera matrix
  cv::Mat distCoeff; //distortion coefficients
  Eigen::Matrix3f K_pixel_eig;
};

}; //end reconstruction