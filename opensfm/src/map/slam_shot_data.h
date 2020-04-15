#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <memory>
#include <map/third_party/data/graph_node.h>

namespace map
{
class SLAMShotData
{
public:
  SLAMShotData() = delete;
  SLAMShotData(Shot* shot):graph_node_(std::make_unique<data::graph_node>(shot, false)){}

  std::vector<cv::KeyPoint> undist_keypts_; // undistorted keypoints
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> bearings_;
  std::vector<std::vector<std::vector<size_t>>> keypt_indices_in_cells_;
  const std::unique_ptr<data::graph_node> graph_node_ = nullptr;
};
} // namespace map