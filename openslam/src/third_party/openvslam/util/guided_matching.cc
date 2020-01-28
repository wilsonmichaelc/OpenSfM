#include "guided_matching.h"


namespace guided_matching
{

// inline bool get_cell_indices(camera::base* camera, const cv::KeyPoint& keypt, int& cell_idx_x, int& cell_idx_y) {
//     cell_idx_x = std::round((keypt.pt.x - camera->img_bounds_.min_x_) * camera->inv_cell_width_);
//     cell_idx_y = std::round((keypt.pt.y - camera->img_bounds_.min_y_) * camera->inv_cell_height_);
//     return (0 <= cell_idx_x && cell_idx_x < static_cast<int>(camera->num_grid_cols_)
//             && 0 <= cell_idx_y && cell_idx_y < static_cast<int>(camera->num_grid_rows_));
// }

void
assign_points_to_grid(const GridParameters& params, const Eigen::MatrixXf& undist_keypts, CellIndices& keypt_indices_in_cells)
{
    const size_t num_pts = undist_keypts.rows();
    const size_t num_to_reserve = 0.5 * num_pts / (params.grid_cols*params.grid_rows);
    keypt_indices_in_cells.resize(params.grid_cols);
    for (auto& keypt_indices_in_row : keypt_indices_in_cells) {
        keypt_indices_in_row.resize(params.grid_cols);
        for (auto& keypt_indices_in_cell : keypt_indices_in_row) {
            keypt_indices_in_cell.reserve(num_to_reserve);
        }
    }
    for (size_t idx = 0; idx < num_pts; ++idx) {
        // const auto& keypt = undist_keypts.at(idx);
        const Eigen::Vector2f pt = undist_keypts.block<2,1>(idx,0);
        const int cell_idx_x = std::round((pt[0] - params.img_min_width) * params.inv_cell_width);
        const int cell_idx_y = std::round((pt[1] - params.img_min_height) * params.inv_cell_height);
        if ((0 <= cell_idx_x && cell_idx_x < static_cast<int>(params.grid_cols)
            && 0 <= cell_idx_y && cell_idx_y < static_cast<int>(params.grid_rows)))
        {
            keypt_indices_in_cells.at(cell_idx_x).at(cell_idx_y).push_back(idx);
        }
    }
}
CellIndices 
assign_keypoints_to_grid(const GridParameters& params, const Eigen::MatrixXf& undist_keypts) {
    CellIndices keypt_indices_in_cells;
    assign_points_to_grid(params, undist_keypts, keypt_indices_in_cells);
    return keypt_indices_in_cells;
}
void match_frame_to_frame(){};
void match_points_to_frame(){};
};