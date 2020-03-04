#pragma once

#include <Eigen/Eigen>
#include <opencv2/features2d/features2d.hpp>

#include <unordered_map>
#include <memory>

using ShotId = int;
using PointId = int;
using FeatureId = int;

class Point {
 public:
 private:
  const std::string point_name_;
  const int id_;

  Eigen::Vector3d coordinates_;
  std::map<Shot *, FeatureId> observations_;
};

class Shot {
 public:
 private:
  const std::string image_name_;
  const int id_;

  std::vector<Point*> points_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
  std::map<FeatureId, PointId> observations_;

  ShotCamera *camera_;
  Pose pose_;

  ShotMeasurements shot_measurements_;
  SLAMData slam_data_;
};

struct SLAMData{
};

struct ShotMeasurements{
  Eigen::Vector3d gps_;
  double timestamp_;
};

class ShotCamera {
  Camera camera_;
  const int id_;
  const std::string camera_name_;
};

class Pose {
public:
  Eigen::Vector3d GetOrigin() const;
  Eigen::Matrix3d WorldToCamera() const;
  Eigen::Matrix3d CameraToWorld() const;
private:
  Eigen::Vector3d translation_;
  Eigen::Vector3d rotation_;
};

class ReconstructionManager {
public:

  // Should belong to the manager
  int GetShotIdFromName(const std::string& name)const;
  int GetPointIdFromName(const std::string& name)const;

  ShotCamera* CreateCamera(const int id, const Camera& camera);

  bool UpdateCamera(const int id, const Camera& camera);

  Shot* CreateShot(const int id, const int camera_id,
                   const Eigen::Vector3d& origin,
                   const Eigen::Vector3d& rotation);
  bool UpdateShotPose(const int id, const Eigen::Vector3d& origin,
                      const Eigen::Vector3d& rotation);

  Point* CreatePoint(const int id, const Eigen::Vector3d& position);
  bool UpdatePoint(const int id, const Eigen::Vector3d& position);

  bool AddObservation(const Shot* shot, const Point* point);
  bool RemoveObservation(const Shot* shot, const Point* point);

  std::map<Point*, FeatureId> GetObservationsOfShot(const Shot* shot);
  std::map<Shot*, FeatureId> GetObservationsOfPoint(const Point* point);  

  const std::unordered_map<ShotId, Shot>& GetAllShots() const;
  const std::unordered_map<int, ShotCamera>& GetAllCameras() const;
  const std::unordered_map<PointId, Point>& GetAllPoints() const;
private:
  std::unordered_map<int, ShotCamera > cameras_;
  std::unordered_map<ShotId, Shot > shots_;
  std::unordered_map<PointId, Point > points_;
};


using TrackId = int;

class TracksManager {
  public:
  std::vector<ShotId> GetShotIds();
  std::vector<TrackId> GetTrackIds();

  cv::KeyPoint GetObservation(const ShotId &shot, const TrackId &point);

  // Not sure if we use that
  std::map<PointId, cv::KeyPoint> GetObservationsOfShot(const ShotId& shot);

  // For point triangulation
  std::map<ShotId, cv::KeyPoint> GetObservationsOfPoint(const PointId& point);

  // For shot resection
  std::map<PointId, cv::KeyPoint> GetObservationsOfPointsAtShot(const std::vector<PointId>& points, const ShotId& shot);

  // For pair bootstrapping
  using ShotIdPair = std::pair<ShotId, ShotId>;
  using KeyPointPair = std::pair<cv::KeyPoint, cv::KeyPoint>;
  std::map<ShotIdPair, KeyPointPair>
  GetAllCommonObservations(const ShotId& shot1, const ShotId& shot2);
};


void do_bundle_in_python(metadata, manager) {

    ba = pybundle.BundleAdjuster()

    for camera in manager.GetAllCameras():
        camera_prior = camera_priors[camera.id]
        _add_camera_to_bundle(ba, camera, camera_prior, fix_cameras)

    for shot in manager.GetAllShots():
        // r = shot.pose.rotation ???
        // t = shot.pose.translation ???
        ba.add_shot(shot.id, shot.camera.id, r, t, False)

    for point in manager.GetAllPoints():
        ba.add_point(point.id, point.coordinates, False)

    for shot in manager.GetAllShots():
      for obs in shot.GetObservations():
        point = obs.feature.keypoint
        scale = obs.feature.scale
        ba.add_point_projection_observation(
            shot.id, obs.point->id, point[0], point[1], scale)

    ///// ??? WHERE DO WE PUT THE METADATA ???
    //// ??? Remove it from Shot in let it be passed
    //// a as map like camera_priors ???
    if config['bundle_use_gps']:
        for shot in manager.GetAllShots():
          shot_metadata = metadata[shot.id]
          g = shot_metadata.gps_position /// ??? DIRECT ACCESS ? Or a Get ?
          ba.add_position_prior(shot.id, g[0], g[1], g[2],
                                shot_metadata.gps_dop)

    if config['bundle_use_gcp'] and gcp:
        _add_gcp_to_bundle(ba, gcp, reconstruction.shots)

    align_method = config['align_method']
    if align_method == 'auto':
        align_method = align.detect_alignment_constraints(config, reconstruction, gcp)
    if align_method == 'orientation_prior':
        if config['align_orientation_prior'] == 'vertical':
            for shot_id in reconstruction.shots:
                ba.add_absolute_up_vector(shot_id, [0, 0, -1], 1e-3)
        if config['align_orientation_prior'] == 'horizontal':
            for shot_id in reconstruction.shots:
                ba.add_absolute_up_vector(shot_id, [0, -1, 0], 1e-3)

    ba.set_point_projection_loss_function(config['loss_function'],
                                          config['loss_function_threshold'])
    ba.set_internal_parameters_prior_sd(
        config['exif_focal_sd'],
        config['principal_point_sd'],
        config['radial_distorsion_k1_sd'],
        config['radial_distorsion_k2_sd'],
        config['radial_distorsion_p1_sd'],
        config['radial_distorsion_p2_sd'],
        config['radial_distorsion_k3_sd'])
    ba.set_num_threads(config['processes'])
    ba.set_max_num_iterations(config['bundle_max_iterations'])
    ba.set_linear_solver_type("SPARSE_SCHUR")

    chrono.lap('setup')
    ba.run()
    chrono.lap('run')

    for camera in manager.GetAllCameras():
      c = ba.get_perspective_camera(camera.id)
      camera_geometry = PerspectiveCamera(c.focal, c.k1, c.k2);
      manager.UpdateCamera(camera.id, camera_geometry);

    for shot in manager.GetAllShots():
        s = ba.get_shot(shot.id)
        manager.UpdateShotPose(camera.id,
                                [s.t[0], s.t[1], s.t[2]],
                                [s.r[0], s.r[1], s.r[2]]);

    for point in manager.GetAllPoints():
        p = ba.get_point(point.id)
        ///// ???? WHAT ABOUT REPROJECTION ERRORS ??? ////
        manager.UpdatePoint(camera.id, [p.p[0], p.p[1], p.p[2]], p.reprojection_errors)

    chrono.lap('teardown')

    logger.debug(ba.brief_report())
    report = {
        'wall_times': dict(chrono.lap_times()),
        'brief_report': ba.brief_report(),
    }
    return report
