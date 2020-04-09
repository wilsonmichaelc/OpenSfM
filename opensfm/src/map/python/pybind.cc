#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <glog/logging.h>

#include <map/pose.h>
#include <map/defines.h>
#include <map/map.h>
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

  py::class_<map::Map>(m, "Map")
    .def(py::init())
    .def("number_of_shots", &map::Map::NumberOfShots, "Returns the number of shots")
    .def("number_of_landmarks", &map::Map::NumberOfLandmarks)
    .def("number_of_cameras", &map::Map::NumberOfCameras)
    .def("create_shot_camera", &map::Map::CreateShotCamera, 
      py::arg("cam_id"), 
      py::arg("camera"),
      py::arg("name") = "",
      py::return_value_policy::reference_internal)
    // .def("update_shot_camera", &map::Map::UpdateShotCamera)
    .def("remove_shot_camera", &map::Map::RemoveShotCamera)
    // Landmark
    .def("create_landmark", &map::Map::CreateLandmark,
        py::arg("lm_id"),
        py::arg("global_position"),
        py::arg("name") = "",
        py::return_value_policy::reference_internal)
    .def("update_landmark", &map::Map::UpdateLandmark)
    .def("remove_landmark", py::overload_cast<const map::Landmark* const>(&map::Map::RemoveLandmark))
    .def("remove_landmark", py::overload_cast<const map::LandmarkId>(&map::Map::RemoveLandmark))
    .def("next_unique_landmark_id", &map::Map::GetNextUniqueLandmarkId)
    // Shot
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::CameraId,
                        const std::string&, const map::Pose&>(&map::Map::CreateShot),
      py::arg("shot_id"), 
      py::arg("shot_cam_id"),
      py::arg("name") = "",
      py::arg("pose") = map::Pose(), 
      py::return_value_policy::reference_internal)
    .def("create_shot", 
      py::overload_cast<const map::ShotId, const map::ShotCamera&,
                        const std::string&, const map::Pose&>(&map::Map::CreateShot),
      py::arg("shot_id"),
      py::arg("shot_cam"),
      py::arg("name") = "",
      py::arg("pose") = map::Pose(),
      py::return_value_policy::reference_internal)
    .def("update_shot_pose", &map::Map::UpdateShotPose)
    .def("remove_shot", &map::Map::RemoveShot)
    .def("next_unique_shot_id", &map::Map::GetNextUniqueShotId)

    .def("add_observation", &map::Map::AddObservation)
    .def("remove_observation", &map::Map::RemoveObservation)
    .def("get_all_shots", &map::Map::GetAllShotPointers, py::return_value_policy::reference_internal)
    .def("get_all_cameras", &map::Map::GetAllCameraPointers, py::return_value_policy::reference_internal)
    .def("get_all_landmarks", &map::Map::GetAllLandmarkPointers, py::return_value_policy::reference_internal)
  ;

  py::class_<map::Shot>(m, "Shot")
    .def(py::init<const map::ShotId, const map::ShotCamera&, 
                  const map::Pose&, const std::string&>())
    .def_readonly("id", &map::Shot::id_)
    .def_readonly("name", &map::Shot::name_)
    .def_readonly("slam_data", &map::Shot::slam_data_, py::return_value_policy::reference_internal)
    .def("get_descriptor", &map::Shot::GetDescriptor, py::return_value_policy::reference_internal)
    .def("get_descriptors", &map::Shot::GetDescriptors, py::return_value_policy::reference_internal)
    .def("get_keypoint", &map::Shot::GetKeyPoint, py::return_value_policy::reference_internal)
    .def("get_keypoints", &map::Shot::GetKeyPoints, py::return_value_policy::reference_internal)
    .def("compute_num_valid_pts", &map::Shot::ComputeNumValidLandmarks)
    .def("number_of_keypoints", &map::Shot::NumberOfKeyPoints)
    .def("init_and_take_datastructures", &map::Shot::InitAndTakeDatastructures)
    .def("init_keypts_and_descriptors", &map::Shot::InitKeyptsAndDescriptors)
    .def("undistort_keypts", &map::Shot::UndistortKeypts)
    .def("undistorted_keypts_to_bearings", &map::Shot::UndistortedKeyptsToBearings)
    .def("set_pose", &map::Shot::SetPose)
    .def("get_pose", &map::Shot::GetPose, py::return_value_policy::reference_internal)
  ;

  py::class_<map::SLAMShotData>(m, "SlamShotData")
    .def_readonly("undist_keypts", &map::SLAMShotData::undist_keypts_)
  ;

  py::class_<map::Landmark>(m, "Landmark")
    .def(py::init<const map::LandmarkId&, const Eigen::Vector3d&, const std::string&>())
    .def_readonly("id", &map::Landmark::id_)
    .def_readonly("name", &map::Landmark::name_)
    .def("get_global_pos", &map::Landmark::GetGlobalPos)
    .def("set_global_pos", &map::Landmark::SetGlobalPos)
    .def("is_observed_in_shot", &map::Landmark::IsObservedInShot)
    .def("add_observation", &map::Landmark::AddObservation)
    .def("remove_observation", &map::Landmark::RemoveObservation)
    .def("has_observations", &map::Landmark::HasObservations)
    .def("get_observations", &map::Landmark::GetObservations, py::return_value_policy::reference_internal)
    .def("number_of_observations", &map::Landmark::NumberOfObservations)
  ;

  py::class_<map::ShotCamera>(m, "ShotCamera")
    .def(py::init<const map::Camera&, const map::CameraId, const std::string&>())
    .def_readonly("id", &map::ShotCamera::id_)
    .def_readonly("camera_name", &map::ShotCamera::camera_name_)
  ;

  py::class_<map::Camera>(m, "Camera")
    .def(py::init<const size_t, const size_t, const std::string&>(),
         py::arg("width"), py::arg("height"), py::arg("projection_type"))
  ;

  py::class_<map::BrownPerspectiveCamera, map::Camera>(m, "BrownPerspectiveCamera")
    .def(py::init<const size_t, const size_t, const std::string&,
                  const float, const float, const float, const float,
                  const float, const float, const float, const float, const float>(),
                  py::arg("width"), py::arg("height"), py::arg("projection_type"),
                  py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
                  py::arg("k1"), py::arg("k2"), py::arg("p1"), py::arg("p2"), py::arg("k3"))
  ;
}
