#pragma once
#include <Eigen/Eigen>

namespace map
{

class Pose 
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Pose():
    cam_to_world_(Eigen::Matrix4d::Identity()), world_to_cam_(Eigen::Matrix4d::Identity())
  {

  }
  //4x4 Transformation
  Eigen::Matrix4d WorldToCamera() const { return world_to_cam_; }
  Eigen::Matrix4d CameraToWorld() const { return cam_to_world_; }

  // 3x3 Rotation
  Eigen::Matrix3d RotationWorldToCamera() const { return world_to_cam_.block<3,3>(0,0); }
  Eigen::Matrix3d RotationCameraToWorld() const { return cam_to_world_.block<3,3>(0,0); }
  
  // 3x1 Translation
  Eigen::Vector3d TranslationWorldToCamera() const { return world_to_cam_.block<3,1>(0,3); }
  Eigen::Vector3d TranslationCameraToWorld() const { return cam_to_world_.block<3,1>(0,3); };
  Eigen::Vector3d GetOrigin() const { return TranslationCameraToWorld(); }

  // void SetPose(const Pose& pose)
  // {
  //   world_to_cam_ = pose.WorldToCamera();
  //   cam_to_world_ = pose.CameraToWorld();
  // }
  void SetFromWorldToCamera(const Eigen::Matrix4d& world_to_camera)
  {
    world_to_cam_ = world_to_camera;
    cam_to_world_ = world_to_camera.inverse();
  }
  void SetFromCameraToWorld(const Eigen::Matrix4d& camera_to_world)
  {
    cam_to_world_ = camera_to_world;
    world_to_cam_ = camera_to_world.inverse();

  }
private:
  Eigen::Matrix4d cam_to_world_;
  Eigen::Matrix4d world_to_cam_;
  //Maybe use Sophus to store the minimum representation
};
}; //namespace map