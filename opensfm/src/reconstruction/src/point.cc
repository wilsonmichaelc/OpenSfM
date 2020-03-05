#include <algorithm>
#include "reconstruction/point.h"
#include "reconstruction/shot.h"

namespace reconstruction
{
Point::Point(const PointId point_id, const Eigen::Vector3d& global_pos, const std::string& name):
  id_(point_id), point_name_(name), global_pos_(global_pos)
{

}

// bool 
// Point::IsObservedInShot(Shot* shot) const
// {
//   return observations_.count(shot);
// }

// void 
// Point::AddObservation(Shot* shot, const FeatureId feat_id)
// {
//   observations_.emplace(shot, feat_id);
// }

// bool
// Point::HasObservations() const
// {
//   return !observations_.empty();
// }

// void
// Point::RemoveObservation(Shot* shot)
// {
//   observations_.erase(shot);
// }
}; //namespace reconstruction
