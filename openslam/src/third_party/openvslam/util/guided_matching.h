#pragma once
#include <Eigen/Eigen>
#include <vector>
namespace guided_matching
{
struct GridParameters
{
    GridParameters(unsigned int grid_cols_, unsigned int grid_rows_,
                   unsigned int img_width_, unsigned int img_height_,
                   unsigned int cell_width_, unsigned int cell_height_)
    {
        grid_cols = grid_cols_; grid_rows = grid_rows;
        img_width = img_width_; img_height = img_height_;
        cell_width = cell_width_; cell_height = cell_height_;
        inv_cell_width = cell_width == 0 ? 0 : 1.0f/float(cell_width);
        inv_cell_height = cell_height == 0 ? 0 : 1.0f/float(cell_height);
    }
    unsigned int grid_cols, grid_rows;
    unsigned int img_width, img_height;
    float cell_width, cell_height;
    float inv_cell_width, inv_cell_height;
};
using CellIndices = std::vector<std::vector<std::vector<unsigned int>>>;

void assign_points_to_grid(const GridParameters& params, const Eigen::MatrixXd& undist_keypts, CellIndices& keypt_indices_in_cells);
CellIndices assign_keypoints_to_grid(const GridParameters& params, const Eigen::MatrixXd& undist_keypts);
void match_frame_to_frame();
void match_points_to_frame();
};