#pragma once
#include <Eigen/Eigen>
#include <unordered_map>
#include <map>
#include <memory>

#include "reconstruction/defines.h"

namespace reconstruction
{
class Shot;
class Landmark;
class Pose;
class ShotCamera;
class Camera;

class ReconstructionManager 
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Should belong to the manager
  ShotId GetShotIdFromName(const std::string& name) const { return shot_names_.at(name); }
  LandmarkId GetPointIdFromName(const std::string& name) const { return landmark_names_.at(name); };

  // Map information
  auto NumberOfShots() const { return shots_.size(); }
  auto NumberOfPoints() const { return landmarks_.size(); }
  auto NumberOfCameras() const { return cameras_.size(); }

  // Create, Update and Remove
  // Camera
  ShotCamera* CreateCamera(const CameraId cam_id, const Camera& camera);
  void UpdateCamera(const CameraId cam_id, const Camera& camera);
  // Shots
  Shot* CreateShot(const ShotId shot_id, const CameraId camera_id, const Pose& pose, const std::string& name = "");
  void UpdateShotPose(const ShotId shot_id, const Pose& pose);
  void RemoveShot(const ShotId shot_id);
  // Point
  Landmark* CreateLandmark(const LandmarkId point_id, const Eigen::Vector3d& global_pos, const std::string& name = "");
  void UpdateLandmark(const LandmarkId point_id, const Eigen::Vector3d& global_pos);
  void RemoveLandmark(const LandmarkId shot_id);

  void AddObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const;
  // (Shot* const shot, const Landmark* point, const FeatureId feat_id);
  void RemoveObservation(Shot *const shot,  Landmark *const lm, const FeatureId feat_id) const;
  // (const Shot* shot, const Landmark* point, const FeatureId feat_id);

  std::map<Landmark*, FeatureId> GetObservationsOfShot(const Shot* shot);
  std::map<Shot*, FeatureId> GetObservationsOfPoint(const Landmark* point);  

  // const std::unordered_map<ShotId, std::unique_ptr<Shot>>& GetAllShots() const { return shots_; }
  // const std::unordered_map<int, std::unique_ptr<ShotCamera>>& GetAllCameras() const { return cameras_; };
  // const std::unordered_map<LandmarkId, std::unique_ptr<Point>>& GetAllPoints() const { return landmarks_; };
  const auto& GetAllShots() const { return shots_; }
  const auto& GetAllCameras() const { return cameras_; };
  const auto& GetAllPoints() const { return landmarks_; };
private:
  std::unordered_map<CameraId, std::unique_ptr<ShotCamera> > cameras_;
  std::unordered_map<ShotId, std::unique_ptr<Shot> > shots_;
  std::unordered_map<LandmarkId, std::unique_ptr<Landmark> > landmarks_;

  std::unordered_map<std::string, ShotId> shot_names_;
  std::unordered_map< std::string, LandmarkId> landmark_names_;
};

} // namespace reconstruction