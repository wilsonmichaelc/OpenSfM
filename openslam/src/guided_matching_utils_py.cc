#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "third_party/openvslam/util/guided_matching.h"
namespace py = pybind11;


PYBIND11_MODULE(guided_matching, m) {
    py::class_<guided_matching::GridParameters>(m, "GridParameters")
    .def(py::init<const unsigned int, const unsigned int, const unsigned int,
                  const unsigned int, const unsigned int, const unsigned int>());
                  
    m.def("assign_points_to_grid", &guided_matching::assign_keypoints_to_grid);
    m.def("match_frame_to_frame", &guided_matching::match_frame_to_frame);
    m.def("match_points_to_frame", &guided_matching::match_points_to_frame);
}

// PYBIND11_MODULE(orb_extractor, m) {

// py::class_<openvslam::feature::orb_extractor>(m, "orb_extractor")
//     .def(py::init<const unsigned int, const float, const unsigned int,
//                   const unsigned int, const unsigned int>())
//     .def("extract_orb_py", &openvslam::feature::orb_extractor::extract_orb_py);
// }