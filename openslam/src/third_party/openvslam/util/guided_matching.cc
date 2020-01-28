#include "guided_matching.h"


namespace guided_matching
{


void
assign_points_to_grid(const GridParameters& params, const Eigen::MatrixXd& undist_keypts, CellIndices& keypt_indices_in_cells)
{

}
CellIndices 
assign_keypoints_to_grid(const GridParameters& params, const Eigen::MatrixXd& undist_keypts) {
    CellIndices keypt_indices_in_cells;
    assign_points_to_grid(params, undist_keypts, keypt_indices_in_cells);
    return keypt_indices_in_cells;
}
};