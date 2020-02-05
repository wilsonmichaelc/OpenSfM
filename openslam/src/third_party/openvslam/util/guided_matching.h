#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include "third_party/openvslam/feature/orb_params.h"
namespace cslam{ 
    class Frame;
    class Landmark;
}
namespace guided_matching
{
struct GridParameters
{
    GridParameters(unsigned int grid_cols_, unsigned int grid_rows_,
                   float img_min_width_, float img_min_height_,
                   float inv_cell_width_, float inv_cell_height_);
    unsigned int grid_cols, grid_rows;
    float img_min_width, img_min_height;
    float inv_cell_width, inv_cell_height;
};
using CellIndices = std::vector<std::vector<std::vector<unsigned int>>>;
using MatchIndices = std::vector<std::pair<size_t, size_t>>;
using OrbFeature = Eigen::Matrix<float, 1, 5>;
void assign_points_to_grid(const GridParameters& params, const Eigen::MatrixXf& undist_keypts, CellIndices& keypt_indices_in_cells);
CellIndices assign_keypoints_to_grid(const GridParameters& params, const Eigen::MatrixXf& undist_keypts);

void distribute_keypoints_to_grid_frame(const GridParameters& params, cslam::Frame& frame);

void distribute_keypoints_to_grid(const GridParameters& params, const std::vector<cv::KeyPoint>& undist_keypts,
                                  CellIndices& keypt_indices_in_cells);

std::vector<size_t> 
get_keypoints_in_cell(const GridParameters& grid_params, const Eigen::MatrixXf& undist_keypts,
                      const CellIndices& keypt_indices_in_cells,
                      const float ref_x, const float ref_y, const float margin,
                      const int min_level, const int max_level);

std::vector<size_t> 
get_keypoints_in_cell(const GridParameters& grid_params, const std::vector<cv::KeyPoint>& undist_keypts,
                      const CellIndices& keypt_indices_in_cells,
                      const float ref_x, const float ref_y, const float margin,
                      const int min_level, const int max_level);

MatchIndices 
match_frame_to_frame_py(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
                    //  const Eigen::MatrixXi& desc_1, const Eigen::MatrixXi& desc_2, 
                    //  const Eigen::MatrixX& desc_1, const Eigen::MatrixXf& desc_2, 
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
                     const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
                     const GridParameters& grid_params, const size_t margin);

std::vector<size_t> 
match_frame_to_frame_dbg(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
                    //  const Eigen::MatrixXi& desc_1, const Eigen::MatrixXi& desc_2, 
                    //  const Eigen::MatrixX& desc_1, const Eigen::MatrixXf& desc_2, 
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
                     const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
                     const GridParameters& grid_params, const size_t margin);
MatchIndices 
match_frame_to_frame(const cslam::Frame& frame1, const cslam::Frame& frame2,
                     const Eigen::MatrixX2f& prevMatched,
                     const GridParameters& grid_params, const size_t margin);


// TODO: Think about the margin. Maybe make it dynamic depending on the depth of the feature!!
size_t
match_frame_and_landmarks(const GridParameters& grid_params, const std::vector<float>& scale_factors, //const openvslam::feature::orb_params& orb_params,
                          cslam::Frame& frame, std::vector<cslam::Landmark*>& local_landmarks, const float margin);
};