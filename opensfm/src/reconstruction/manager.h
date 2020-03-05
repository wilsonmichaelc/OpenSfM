#pragma once
#include <Eigen/Eigen>
#include <unordered_map>
#include <map>
#include <memory>

#include "reconstruction/defines.h"

namespace reconstruction
{
class Shot;
class Point;
class Pose;
class ShotCamera;
class Camera;

class ReconstructionManager 
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Should belong to the manager
  ShotId GetShotIdFromName(const std::string& name) const { return shot_names_.at(name); }
  PointId GetPointIdFromName(const std::string& name) const { return point_names_.at(name); };

  // Map information
  auto NumberOfShots() const { return shots_.size(); }
  auto NumberOfPoints() const { return points_.size(); }
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
  Point* CreatePoint(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name = "");
  void UpdatePoint(const PointId point_id, const Eigen::Vector3d& global_pos);
  void RemovePoint(const PointId shot_id);

  void AddObservation(Shot *const shot,  Point *const point, const FeatureId feat_id) const;
  // (Shot* const shot, const Point* point, const FeatureId feat_id);
  void RemoveObservation(Shot *const shot,  Point *const point, const FeatureId feat_id) const;
  // (const Shot* shot, const Point* point, const FeatureId feat_id);

  std::map<Point*, FeatureId> GetObservationsOfShot(const Shot* shot);
  std::map<Shot*, FeatureId> GetObservationsOfPoint(const Point* point);  

  // const std::unordered_map<ShotId, std::unique_ptr<Shot>>& GetAllShots() const { return shots_; }
  // const std::unordered_map<int, std::unique_ptr<ShotCamera>>& GetAllCameras() const { return cameras_; };
  // const std::unordered_map<PointId, std::unique_ptr<Point>>& GetAllPoints() const { return points_; };
  const auto& GetAllShots() const { return shots_; }
  const auto& GetAllCameras() const { return cameras_; };
  const auto& GetAllPoints() const { return points_; };
private:
  std::unordered_map<CameraId, std::unique_ptr<ShotCamera> > cameras_;
  std::unordered_map<ShotId, std::unique_ptr<Shot> > shots_;
  std::unordered_map<PointId, std::unique_ptr<Point> > points_;

  std::unordered_map<std::string, ShotId> shot_names_;
  std::unordered_map< std::string, PointId> point_names_;
};

} // namespace reconstruction