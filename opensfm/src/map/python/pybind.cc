#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <glog/logging.h>

#include <map/pose.h>
#include <map/defines.h>
#include <map/manager.h>
#include <map/shot.h>
#include <map/landmark.h>
#include <map/camera.h>

namespace py = pybind11;
PYBIND11_MODULE(pymap, m) {

  py::class_<map::Pose>(m, "Pose")
    .def(py::init())
    .def("get_cam_to_world", &map::Pose::CameraToWorld)
    .def("get_world_to_cam", &map::Pose::WorldToCamera)
    .def("set_from_cam_to_world", &map::Pose::SetFromCameraToWorld)
    .def("set_from_world_to_cam", &map::Pose::SetFromWorldToCamera)
  ;

  py::class_<map::Manager>(m, "Manager")
    .def(py::init())
    .def("number_of_shots", &map::Manager::NumberOfShots)
    .def("number_of_landmarks", &map::Manager::NumberOfLandmarks)
    .def("number_of_cameras", &map::Manager::NumberOfCameras)
    .def("create_camera", &map::Manager::CreateShotCamera)
    .def("remove_camera", &map::Manager::RemoveShotCamera)
    .def("create_landmark", &map::Manager::CreateLandmark)
    .def("update_landmark", &map::Manager::UpdateLandmark)
    .def("remove_landmark", &map::Manager::RemoveLandmark)
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::CameraId,
                        const map::Pose&, const std::string&>(&map::Manager::CreateShot),
      py::arg("shot_id"), py::arg("cam_id"), py::arg("pose") = map::Pose(), py::arg("name") = "")
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::ShotCamera&,
                        const map::Pose&, const std::string&>(&map::Manager::CreateShot),
      py::arg("shot_id"), py::arg("cam_id"), py::arg("pose") = map::Pose(), py::arg("name") = "")
    .def("update_shot_pose", &map::Manager::UpdateShotPose)
    .def("remove_shot", &map::Manager::RemoveShot)
    .def("add_observation", &map::Manager::AddObservation)
    .def("remove_observation", &map::Manager::RemoveObservation)
    // .def("get_all_shots", &map::Manager::GetAllShots)
    // .def("get_all_camera", &map::Manager::GetAllCameras)
    // .def("get_all_landmarks", &map::Manager::GetAllLandmarks)
    .def("get_all_shots", &map::Manager::GetAllShotPointers)
    .def("get_all_cameras", &map::Manager::GetAllCameraPointers)
    .def("get_all_landmarks", &map::Manager::GetAllLandmarkPointers)
  ;

  py::class_<map::Shot>(m, "Shot")
    .def(py::init<const map::ShotId, const map::ShotCamera&, 
                  const map::Pose&, const std::string&>())
    .def_readonly("id_", &map::Shot::id_)
    .def_readonly("shot_name_", &map::Shot::shot_name_)
    .def("get_descriptor", &map::Shot::GetDescriptor)
    .def("get_descriptors", &map::Shot::GetDescriptors)
    .def("get_keypoint", &map::Shot::GetKeyPoint)
    .def("get_keypoints", &map::Shot::GetKeyPoints)
    .def("compute_num_valid_pts", &map::Shot::ComputeNumValidLandmarks)
    .def("init_and_take_datastructures", &map::Shot::InitAndTakeDatastructures)
  ;

  py::class_<map::Landmark>(m, "Landmark")
    .def(py::init<const map::LandmarkId&, const Eigen::Vector3d&, const std::string&>())
    .def_readonly("id_", &map::Landmark::id_)
    .def_readonly("point_name_", &map::Landmark::point_name_)
    .def("get_global_pos", &map::Landmark::GetGlobalPos)
    .def("set_global_pos", &map::Landmark::SetGlobalPos)
    .def("is_observed_in_Shot", &map::Landmark::IsObservedInShot)
    .def("add_observation", &map::Landmark::AddObservation)
    .def("remove_observation", &map::Landmark::RemoveObservation)
    .def("has_observations", &map::Landmark::HasObservations)
    .def("get_observation", &map::Landmark::GetObservations)
  ;

  py::class_<map::ShotCamera>(m, "ShotCamera")
    .def(py::init<const map::Camera&, const map::CameraId, const std::string&>())
    .def_readonly("id_", &map::ShotCamera::id_)
    .def_readonly("camera_name_", &map::ShotCamera::camera_name_)
  ;

  py::class_<map::Camera>(m, "Camera")
    .def(py::init())
  ;
}
