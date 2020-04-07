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
    .def("number_of_shots", &map::Manager::NumberOfShots, "Returns the number of shots")
    .def("number_of_landmarks", &map::Manager::NumberOfLandmarks)
    .def("number_of_cameras", &map::Manager::NumberOfCameras)
    .def("create_shot_camera", &map::Manager::CreateShotCamera, py::return_value_policy::reference_internal)
    .def("remove_shot_camera", &map::Manager::RemoveShotCamera)
    .def("create_landmark", &map::Manager::CreateLandmark,
        py::arg("lm_id"),
        py::arg("global_position"),
        py::arg("name") = "",
        py::return_value_policy::reference_internal)
    .def("update_landmark", &map::Manager::UpdateLandmark)
    .def("remove_landmark", py::overload_cast<const map::Landmark* const>(&map::Manager::RemoveLandmark))
    .def("remove_landmark", py::overload_cast<const map::LandmarkId>(&map::Manager::RemoveLandmark))
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::CameraId,
                        const map::Pose&, const std::string&>(&map::Manager::CreateShot),
      py::arg("shot_id"), 
      py::arg("cam_id"),
      py::arg("pose") = map::Pose(), 
      py::arg("name") = "",
      py::return_value_policy::reference_internal)
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::ShotCamera&,
                        const map::Pose&, const std::string&>(&map::Manager::CreateShot),
      py::arg("shot_id"),
      py::arg("cam_id"),
      py::arg("pose") = map::Pose(),
      py::arg("name") = "",
      py::return_value_policy::reference_internal)
    .def("update_shot_pose", &map::Manager::UpdateShotPose)
    .def("remove_shot", &map::Manager::RemoveShot)
    .def("add_observation", &map::Manager::AddObservation)
    .def("remove_observation", &map::Manager::RemoveObservation)
    .def("get_all_shots", &map::Manager::GetAllShotPointers, py::return_value_policy::reference_internal)
    .def("get_all_cameras", &map::Manager::GetAllCameraPointers, py::return_value_policy::reference_internal)
    .def("get_all_landmarks", &map::Manager::GetAllLandmarkPointers, py::return_value_policy::reference_internal)
  ;

  py::class_<map::Shot>(m, "Shot")
    .def(py::init<const map::ShotId, const map::ShotCamera&, 
                  const map::Pose&, const std::string&>())
    .def_readonly("id_", &map::Shot::id_)
    .def_readonly("name_", &map::Shot::name_)
    .def("get_descriptor", &map::Shot::GetDescriptor, py::return_value_policy::reference_internal)
    .def("get_descriptors", &map::Shot::GetDescriptors, py::return_value_policy::reference_internal)
    .def("get_keypoint", &map::Shot::GetKeyPoint, py::return_value_policy::reference_internal)
    .def("get_keypoints", &map::Shot::GetKeyPoints, py::return_value_policy::reference_internal)
    .def("compute_num_valid_pts", &map::Shot::ComputeNumValidLandmarks)
    .def("number_of_keypoints", &map::Shot::NumberOfKeyPoints)
    .def("init_and_take_datastructures", &map::Shot::InitAndTakeDatastructures)
    .def("init_keypts_and_descriptors", &map::Shot::InitKeyptsAndDescriptors)
  ;

  py::class_<map::Landmark>(m, "Landmark")
    .def(py::init<const map::LandmarkId&, const Eigen::Vector3d&, const std::string&>())
    .def_readonly("id_", &map::Landmark::id_)
    .def_readonly("name_", &map::Landmark::name_)
    .def("get_global_pos", &map::Landmark::GetGlobalPos)
    .def("set_global_pos", &map::Landmark::SetGlobalPos)
    .def("is_observed_in_Shot", &map::Landmark::IsObservedInShot)
    .def("add_observation", &map::Landmark::AddObservation)
    .def("remove_observation", &map::Landmark::RemoveObservation)
    .def("has_observations", &map::Landmark::HasObservations)
    .def("get_observations", &map::Landmark::GetObservations, py::return_value_policy::reference_internal)
    .def("number_of_observations", &map::Landmark::NumberOfObservations)
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
