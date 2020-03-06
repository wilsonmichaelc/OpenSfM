#pragma once
#include <Eigen/Eigen>
#include <unordered_map>
#include <map>
#include <memory>

#include <map/defines.h>
#include <map/pose.h>
namespace map
{
class Shot;
class Landmark;
class ShotCamera;
class Camera;

class Manager 
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Should belong to the manager
  ShotId GetShotIdFromName(const std::string& name) const { return shot_names_.at(name); }
  LandmarkId GetPointIdFromName(const std::string& name) const { return landmark_names_.at(name); };

  // Map information
  auto NumberOfShots() const { return shots_.size(); }
  auto NumberOfLandmarks() const { return landmarks_.size(); }
  auto NumberOfCameras() const { return cameras_.size(); }

  // Create, Update and Remove

  // Camera
  ShotCamera* CreateShotCamera(const CameraId cam_id, const Camera& camera, const std::string& name = "");
  void UpdateCamera(const CameraId cam_id, const Camera& camera);
  void RemoveShotCamera(const CameraId cam_id);

  // Shots
  Shot* CreateShot(const ShotId shot_id, const CameraId camera_id, const Pose& pose = Pose(), const std::string& name = "");
  Shot* CreateShot(const ShotId shot_id, const ShotCamera& shot_cam, const Pose& pose = Pose(), const std::string& name = "");
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
  // const std::unordered_map<CameraId, std::unique_ptr<ShotCamera>>& GetAllCameras() const { return cameras_; };
  // const std::unordered_map<LandmarkId, std::unique_ptr<Landmark>>& GetAllLandmarks() const { return landmarks_; };
  const auto& GetAllShots() const { return shots_; }
  const auto& GetAllCameras() const { return cameras_; };
  const auto& GetAllLandmarks() const { return landmarks_; };

  const auto GetAllShotPointers() const
  {
    std::unordered_map<ShotId, Shot*> copy;
    std::transform(shots_.begin(), shots_.end(), std::inserter(copy, copy.end()), [](auto& elem) { return std::make_pair(elem.first, elem.second.get()); });
    return copy;
  }
  const auto GetAllCameraPointers() const
  {
    std::unordered_map<CameraId, ShotCamera*> copy;
    std::transform(cameras_.begin(), cameras_.end(), std::inserter(copy, copy.end()), [](auto& elem) { return std::make_pair(elem.first, elem.second.get()); });
    return copy;
  }
  const auto GetAllLandmarkPointers() const
  {
    std::unordered_map<LandmarkId, Landmark*> copy;
    std::transform(landmarks_.begin(), landmarks_.end(), std::inserter(copy, copy.end()), [](auto& elem) { return std::make_pair(elem.first, elem.second.get()); });
    return copy;
  }
private:
  std::unordered_map<CameraId, std::unique_ptr<ShotCamera> > cameras_;
  std::unordered_map<ShotId, std::unique_ptr<Shot> > shots_;
  std::unordered_map<LandmarkId, std::unique_ptr<Landmark> > landmarks_;

  std::unordered_map<std::string, ShotId> shot_names_;
  std::unordered_map< std::string, LandmarkId> landmark_names_;
  std::unordered_map<std::string, CameraId> camera_names_;
};

} // namespace map