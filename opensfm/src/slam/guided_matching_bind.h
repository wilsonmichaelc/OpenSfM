#pragma once
#include <slam/guided_matching.h>
#include <iostream>
namespace slam
{
class GuidedMatchingWrapper
{
public:
  GuidedMatchingWrapper(const GridParameters& grid_params):matcher_(grid_params) {}
  void DistributeUndistKeyptsToGrid(map::Shot& shot)
  {
    matcher_.DistributeUndistKeyptsToGrid(shot.slam_data_.undist_keypts_, shot.slam_data_.keypt_indices_in_cells_);
  }

  MatchIndices
  MatchShotToShot(map::Shot& shot1, map::Shot& shot2, const Eigen::MatrixX2f& prev_matched, const size_t margin) const
  {
    std::cout << "kpts:" << shot1.slam_data_.undist_keypts_.size() << "/" << shot2.slam_data_.undist_keypts_.size()
              << "cells: " << shot2.slam_data_.keypt_indices_in_cells_.size();
              //  << ", prev: " << prev_matched << "margin" << margin << std::endl;
    return matcher_.MatchKptsToKpts(shot1.slam_data_.undist_keypts_, shot1.GetDescriptors(),
                                    shot2.slam_data_.undist_keypts_, shot2.GetDescriptors(),
                                    shot2.slam_data_.keypt_indices_in_cells_,
                                    prev_matched, margin);
  }
private:
  GuidedMatcher matcher_;
};
} // namespace slam
