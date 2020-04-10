#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <glog/logging.h>
#include <slam/orb_extractor_bind.h>
#include <slam/guided_matching_bind.h>
// #include <slam/guided_matching.h>
#include <slam/slam_utilities.h>
#include <slam/pyslam_utilities.h>
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
    .def("get_scale_levels", &slam::OrbExtractorWrapper::GetScaleLevels)//, py::return_value_policy::reference_internal)
  ;
  
  py::class_<slam::GuidedMatchingWrapper>(m, "GuidedMatcher")
    .def(py::init<const slam::GridParameters&>(), py::arg("grid_parameters"))
    .def("distribute_undist_keypts_to_grid", &slam::GuidedMatchingWrapper::DistributeUndistKeyptsToGrid,
         py::arg("shot"))
    .def("match_shot_to_shot", &slam::GuidedMatchingWrapper::MatchShotToShot,
         py::arg("shot1"), py::arg("shot2"), py::arg("prev_matched"), py::arg("margin"))
  ;

  // Helper class
  py::class_<slam::GridParameters>(m, "GridParameters")
    .def(py::init<unsigned int, unsigned int, float, float, float, float, float, float>())
  ;
  
  // py::class_<slam::SlamUtilities>(m, "SlamUtilities")
  //   .def("convert_keypts_to_eigen", &slam::SlamUtilities::ConvertOpenCVKptsToEigen)
  // ;

  py::class_<slam::PySlamUtilities>(m, "SlamUtilities")
    .def("undist_keypts_from_shot", &slam::PySlamUtilities::GetUndistortedKeyptsFromShot)
    .def("keypts_from_shot", &slam::PySlamUtilities::GetKeyptsFromShot)
    .def("compute_descriptor", &slam::PySlamUtilities::SetDescriptorFromObservations)
    .def("compute_normal_and_depth", &slam::PySlamUtilities::SetNormalAndDepthFromObservations)
  ;

}
