
#include <slam/slam_utilities.h>

namespace slam
{

Eigen::Matrix3f
SlamUtilities::to_skew_symmetric_mat(const Eigen::Vector3f &vec)
{
  Eigen::Matrix3f skew;
  skew << 0, -vec(2), vec(1),
      vec(2), 0, -vec(0),
      -vec(1), vec(0), 0;
  return skew;
}
Eigen::Matrix3f
SlamUtilities::create_E_21(const Eigen::Matrix3f &rot_1w, const Eigen::Vector3f &trans_1w,
                           const Eigen::Matrix3f &rot_2w, const Eigen::Vector3f &trans_2w)
{
  const Eigen::Matrix3f rot_21 = rot_2w * rot_1w.transpose();
  const Eigen::Vector3f trans_21 = -rot_21 * trans_1w + trans_2w;
  const Eigen::Matrix3f trans_21_x = to_skew_symmetric_mat(trans_21);
  return trans_21_x * rot_21;
}

bool
SlamUtilities::check_epipolar_constraint(const Eigen::Vector3f &bearing_1, const Eigen::Vector3f &bearing_2,
                                         const Eigen::Matrix3f &E_12, const float bearing_1_scale_factor)
{
  // keyframe1上のtエピポーラ平面の法線ベクトル
  const Eigen::Vector3f epiplane_in_1 = E_12 * bearing_2;

  // 法線ベクトルとbearingのなす角を求める
  const auto cos_residual = epiplane_in_1.dot(bearing_1) / epiplane_in_1.norm();
  const auto residual_rad = M_PI / 2.0 - std::abs(std::acos(cos_residual));

  // inlierの閾値(=0.2deg)
  // (e.g. FOV=90deg,横900pixのカメラにおいて,0.2degは横方向の2pixに相当)
  // TODO: 閾値のパラメータ化
  constexpr double residual_deg_thr = 0.2;
  constexpr double residual_rad_thr = residual_deg_thr * M_PI / 180.0;

  // 特徴点スケールが大きいほど閾値を緩くする
  // TODO: thresholdの重み付けの検討
  return residual_rad < residual_rad_thr * bearing_1_scale_factor;
}

Eigen::MatrixXf 
SlamUtilities::convertOpenCVKptsToEigen(const std::vector<cv::KeyPoint>& keypts)
{
  if (!keypts.empty())
  {
    const auto n_kpts = keypts.size();
    Eigen::MatrixXf mat(n_kpts, 5);
    for (size_t i = 0; i < n_kpts; ++i)
    {
      const auto& kpt = keypts[i];
      mat.row(i) << kpt.pt.x, kpt.pt.y, kpt.size, kpt.angle, kpt.octave;
    }
    return mat;
  }
  return Eigen::MatrixXf();
}


} // namespace map