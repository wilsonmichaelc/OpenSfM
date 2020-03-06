#pragma once
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Eigen>

#include <map/defines.h>
#include <map/pose.h>

namespace map
{
class Pose;
class Camera;
class Landmark;

struct ShotCamera {
  ShotCamera(const Camera& camera, const CameraId cam_id, const std::string cam_name = ""):
    camera_(camera), id_(cam_id), camera_name_(cam_name){}
    const Camera& camera_;

    const int id_;
    const std::string camera_name_;
};
struct SLAMShotData
{

};

struct ShotMeasurements
{
  Eigen::Vector3d gps_;
  double timestamp_;
};

class Shot {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Shot(const ShotId shot_id, const ShotCamera& camera, const Pose& pose, const std::string& name = "");
  const cv::Mat GetDescriptor(const FeatureId id) const { return descriptors_.row(id); }
  const cv::KeyPoint& GetKeyPoint(const FeatureId id) const { return keypoints_.at(id); }
  
  //No reason to set individual keypoints or descriptors
  //read-only access
  const std::vector<cv::KeyPoint>& GetKeyPoints() const { return keypoints_; }
  const cv::Mat& GetDescriptors() const { return descriptors_; }
  
  size_t ComputeNumValidLandmarks() const;
  
  const std::vector<Landmark*>& GetPoints() const { return landmarks_; }
  std::vector<Landmark*>& GetPoints() { return landmarks_; }
  void RemoveLandmarkObservation(const FeatureId id);
  void AddPointObservation(Landmark* lm, const FeatureId feat_id) { landmarks_.at(feat_id) = lm; }
  void SetPose(const Pose& pose) { pose_ = pose; }
  SLAMShotData slam_data_;
  void InitAndTakeDatastructures(std::vector<cv::KeyPoint> keypts, cv::Mat descriptors);

  // void SetAndTakeKeyPoints(std::vector<cv::KeyPoint> keypts){ std::swap(keypts, keypoints_); }
  // void SetAndTakeDescriptors(cv::Mat descriptors) { std::swap(descriptors, descriptors_); }

public:
  //We could set the const values to public, to avoid writing a getter.
  const ShotId id_;
  const std::string shot_name_;

 private:
  // const ShotCamera *camera_;
  const ShotCamera& camera_;
  Pose pose_;
  size_t num_keypts_;

  std::vector<Landmark*> landmarks_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;

  ShotMeasurements shot_measurements_;

};
}