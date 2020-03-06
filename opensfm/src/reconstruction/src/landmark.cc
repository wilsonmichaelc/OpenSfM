#include <algorithm>
#include "reconstruction/landmark.h"
#include "reconstruction/shot.h"

namespace reconstruction
{
Landmark::Landmark(const LandmarkId point_id, const Eigen::Vector3d& global_pos, const std::string& name):
  id_(point_id), point_name_(name), global_pos_(global_pos)
{

}
}; //namespace reconstruction
