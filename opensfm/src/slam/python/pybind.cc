#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <glog/logging.h>
#include <slam/orb_extractor_bind.h>
namespace py = pybind11;

PYBIND11_MODULE(pyslam, m) {

  py::class_<slam::OrbExtractorWrapper>(m, "OrbExtractor")
    .def(py::init<const unsigned int, const float, const unsigned int,
                  const unsigned int, const unsigned int>(),
        py::arg("max_kpts"),
        py::arg("scale_factor"),
        py::arg("num_levels"),
        py::arg("ini_fast_thr"),
        py::arg("min_fast_thr"))
    .def("extract_to_shot", &slam::OrbExtractorWrapper::extract_to_shot)
    .def("extract", &slam::OrbExtractorWrapper::extract)
  ;
  
}
