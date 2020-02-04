// #pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "slam_datastructures/frame.h"
#include "slam_datastructures/keyframe.h"
#include "slam_datastructures/landmark.h"
#include "slam_datastructures/camera.h"

namespace py = pybind11;
// Define the bindings!
PYBIND11_MODULE(cslam_types, m) {
    // Frame
    py::class_<cslam::Frame>(m, "Frame")
        .def(py::init< const std::string&, const size_t>())
        .def_readonly("frame_id", &cslam::Frame::mFrameId)
        .def_readonly("im_name", &cslam::Frame::mImgName)
        // .def_readonly("keypts", &cslam::Frame::mKeyPts)
        .def("getKptsAndDescPy",&cslam::Frame::getKptsAndDescPy)
        .def("getKptsPy",&cslam::Frame::getKptsPy)
        .def("getDescPy",&cslam::Frame::getDescPy)
        .def("getKptsUndist", &cslam::Frame::getKptsUndist);
    // Keyframe
    // Landmark

    //Camera
    py::class_<cslam::BrownPerspectiveCamera>(m, "BrownPerspectiveCamera")
        .def(py::init<const size_t, const size_t, const std::string&,
                      const float, const float, const float, const float,
                      const float, const float, const float, const float, const float>())
        .def("undistKeyptsFrame", &cslam::BrownPerspectiveCamera::undistKeyptsFrame);


        // (const size_t width_, const size_t height_, const std::string& projection_type_,
        //  const float fx_, const float fy_, const float cx_, const float cy_,
        //  const float k1_, const float k2_, const float p1_, const float p2_, const float k3_)
}