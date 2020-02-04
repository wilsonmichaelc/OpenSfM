#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "openvslam/feature/orb_extractor.h"


namespace py = pybind11;


PYBIND11_MODULE(orb_extractor, m) {

py::class_<openvslam::feature::orb_extractor>(m, "orb_extractor")
    .def(py::init<const unsigned int, const float, const unsigned int,
                  const unsigned int, const unsigned int>())
    .def("extract_orb_py", &openvslam::feature::orb_extractor::extract_orb_py)
    .def("extract_orb_py2", &openvslam::feature::orb_extractor::extract_orb_py2);
}

