#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "third_party/openvslam/util/guided_matching.h"
namespace py = pybind11;


PYBIND11_MODULE(orb_extractor, m) {
    m.def("assign_points_to_grid", &guided_matching::assign_keypoints_to_grid);
    m.def("match_frame_to_frame", &guided_matching::match_frame_to_frame);
    m.def("match_points_to_frame", &guided_matching::match_points_to_frame);
}