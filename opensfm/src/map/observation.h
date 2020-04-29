#pragma once
#include <Eigen/Core>

namespace map
{
// TODO: unify with sfm Observation
struct Observation
{
  Observation() = default;
  Observation(double x, double y, double s, int r, int g, int b, int id)
      : point(x, y), scale(s), color(r, g, b), id(id)
        angle(0), response(0), octave(0), class_id(-1)
        { }
  bool operator==(const Observation &k) const
  {
    return point == k.point && scale == k.scale && color == k.color &&
           id == k.id;
  }
  Eigen::Vector2d point;
  double scale{1};
  Eigen::Matrix<uint8_t, 3, 1> color;
  int id{0};

  float angle;
  float response;
  int octave;
  int class_id;
};
} // namespace map