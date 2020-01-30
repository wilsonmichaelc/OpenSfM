#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "third_party/openvslam/util/guided_matching.h"
namespace py = pybind11;


PYBIND11_MODULE(guided_matching, m) {
    py::class_<guided_matching::GridParameters>(m, "GridParameters")
    .def(py::init<unsigned int, unsigned int, float, float, float, float>());

    m.def("assign_keypoints_to_grid", &guided_matching::assign_keypoints_to_grid);
    // m.def("match_frame_to_frame", &guided_matching::match_frame_to_frame,
    //     py::arg("undist_kpts1"), py::arg("undist_kpts_2"),
    //     py::arg("desc_1"), py::arg("desc_2"),
    //     py::arg("cell_indices_2"),
    //     py::arg("prev_matched_pts"),
    //     py::arg("grid_params"),
    //     py::arg("margin")
    // );
    m.def("match_frame_to_frame", &guided_matching::match_frame_to_frame);
    m.def("match_frame_to_frame_dbg", &guided_matching::match_frame_to_frame_dbg);

    m.def("match_points_to_frame", 
        &guided_matching::match_points_to_frame);
}
// py::arg("image"),
//         py::arg("peak_threshold") = 0.003,
//         py::arg("edge_threshold") = 10,
//         py::arg("target_num_features") = 0,
//         py::arg("use_adaptive_suppression") = false
// match_frame_to_frame(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
//                      const CellIndices& cell_indices_2, const GridParameters& grid_params,
//                      const Eigen::MatrixX2f& prevMatched, const size_t margin)