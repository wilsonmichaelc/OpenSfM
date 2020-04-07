#pragma once
#include <Eigen/Eigen>
#include <opencv2/features2d/features2d.hpp>
#include <map>
#include <memory>
#include <map/defines.h>
#include <iostream>
namespace map
{
class Shot;

struct SLAMPointData{
};
class Landmark {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Landmark(const LandmarkId point_id, const Eigen::Vector3d& global_pos, const std::string& name = "");
  ~Landmark()
  {
    std::cout << "Deleting lm: " << id_ << std::endl;
  }
  const Eigen::Vector3d& GetGlobalPos() const { return global_pos_; }
  void SetGlobalPos(const Eigen::Vector3d& global_pos) { global_pos_ = global_pos; }

  bool IsObservedInShot(Shot* shot) const { return observations_.count(shot) > 0; }
  void AddObservation(Shot* shot, const FeatureId feat_id) { observations_.emplace(shot, feat_id); }
  void RemoveObservation(Shot* shot) { observations_.erase(shot); }
  bool HasObservations() const { return !observations_.empty(); }
  auto NumberOfObservations() const { return observations_.size(); }
  const auto& GetObservations() const { return observations_; }
public:
  //We could set the const values to public, to avoid writing a getter.
  const LandmarkId id_;
  const std::string name_;
private:
  Eigen::Vector3d global_pos_; // point in global
  std::map<Shot*, FeatureId, KeyCompare> observations_;
  // ,
          //  Eigen::aligned_allocator<std::pair<Shot*, FeatureId>>> observations_;
  SLAMPointData slam_data_;
};
}