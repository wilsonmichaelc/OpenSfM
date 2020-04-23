#pragma once
#include <third_party/json/json.hpp>
using json = nlohmann::json;

namespace map
{
class Map;
class Shot;
class Landmark;

class MapExporter
{
public:
  static void MapToJson(const map::Map& rec_map);
  static void ShotToJson(const map::Shot& shot);
  static void LandmarkToJson(const map::Landmark& landmark);
};
}