#pragma once
#include <Eigen/Core>
#include <vector>
#include <opencv2/core.hpp>

namespace slam
{
class SlamUtilities
{
public:
  static bool check_epipolar_constraint(const Eigen::Vector3f& bearing_1, const Eigen::Vector3f& bearing_2,
                                        const Eigen::Matrix3f& E_12, const float bearing_1_scale_factor);

  static Eigen::Matrix3f to_skew_symmetric_mat(const Eigen::Vector3f& vec);

  
  static Eigen::Matrix3f create_E_21(const Eigen::Matrix3f& rot_1w, const Eigen::Vector3f& trans_1w,
                                     const Eigen::Matrix3f& rot_2w, const Eigen::Vector3f& trans_2w);

  static Eigen::MatrixXf ConvertOpenCVKptsToEigen(const std::vector<cv::KeyPoint>& keypts);
  // static void SlamUtilities::ComputeDescriptor(Landmark& landmark);
};
} // namespace map