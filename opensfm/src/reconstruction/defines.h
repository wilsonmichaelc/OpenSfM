#pragma once
namespace reconstruction
{
struct KeyCompare
{
    template<typename T>
    bool operator()(T* lhs, T* rhs) const { return lhs->id_ < rhs->id_; }
    template<typename T>
    bool operator()(T const* lhs, T const * rhs) const { return lhs->id_ < rhs->id_; }
};
using ShotId = size_t;
using PointId = size_t;
using FeatureId = size_t;
using CameraId = size_t;
}