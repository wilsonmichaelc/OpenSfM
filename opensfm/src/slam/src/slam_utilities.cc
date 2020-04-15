
#include <slam/slam_utilities.h>
#include <map/shot.h>
#include <map/landmark.h>

#include <unordered_set>
#include <unordered_map>
namespace slam
{

Eigen::Matrix3f
SlamUtilities::to_skew_symmetric_mat(const Eigen::Vector3f &vec)
{
  Eigen::Matrix3f skew;
  skew << 0, -vec(2), vec(1),
      vec(2), 0, -vec(0),
      -vec(1), vec(0), 0;
  return skew;
}
Eigen::Matrix3f
SlamUtilities::create_E_21(const Eigen::Matrix3f &rot_1w, const Eigen::Vector3f &trans_1w,
                           const Eigen::Matrix3f &rot_2w, const Eigen::Vector3f &trans_2w)
{
  const Eigen::Matrix3f rot_21 = rot_2w * rot_1w.transpose();
  const Eigen::Vector3f trans_21 = -rot_21 * trans_1w + trans_2w;
  const Eigen::Matrix3f trans_21_x = to_skew_symmetric_mat(trans_21);
  return trans_21_x * rot_21;
}

bool SlamUtilities::check_epipolar_constraint(const Eigen::Vector3f &bearing_1, const Eigen::Vector3f &bearing_2,
                                              const Eigen::Matrix3f &E_12, const float bearing_1_scale_factor)
{
  // keyframe1上のtエピポーラ平面の法線ベクトル
  const Eigen::Vector3f epiplane_in_1 = E_12 * bearing_2;

  // 法線ベクトルとbearingのなす角を求める
  const auto cos_residual = epiplane_in_1.dot(bearing_1) / epiplane_in_1.norm();
  const auto residual_rad = M_PI / 2.0 - std::abs(std::acos(cos_residual));

  // inlierの閾値(=0.2deg)
  // (e.g. FOV=90deg,横900pixのカメラにおいて,0.2degは横方向の2pixに相当)
  // TODO: 閾値のパラメータ化
  constexpr double residual_deg_thr = 0.2;
  constexpr double residual_rad_thr = residual_deg_thr * M_PI / 180.0;

  // 特徴点スケールが大きいほど閾値を緩くする
  // TODO: thresholdの重み付けの検討
  return residual_rad < residual_rad_thr * bearing_1_scale_factor;
}

Eigen::MatrixXf
SlamUtilities::ConvertOpenCVKptsToEigen(const std::vector<cv::KeyPoint> &keypts)
{
  if (!keypts.empty())
  {
    const auto n_kpts = keypts.size();
    Eigen::MatrixXf mat(n_kpts, 5);
    for (size_t i = 0; i < n_kpts; ++i)
    {
      const auto &kpt = keypts[i];
      mat.row(i) << kpt.pt.x, kpt.pt.y, kpt.size, kpt.angle, kpt.octave;
    }
    return mat;
  }
  return Eigen::MatrixXf();
}

std::vector<map::Landmark *>
SlamUtilities::update_local_landmarks(const std::vector<map::Shot *> &local_keyframes)
{
  // std::vector<map::Landmark*> local_landmarks;
  std::unordered_set<map::Landmark *> local_landmarks;
  for (auto keyframe : local_keyframes)
  {
    for (auto lm : keyframe->GetLandmarks())
    {
      if (lm != nullptr)
      {
        // do not add twice
        // if (lm->identifier_in_local_map_update_ == curr_frm_id) continue;
        // lm->identifier_in_local_map_update_ = curr_frm_id;
        local_landmarks.emplace(lm);
      }
    }
  }
  return std::vector<map::Landmark *>(local_landmarks.begin(), local_landmarks.end());
}

std::vector<map::Shot *>
SlamUtilities::update_local_keyframes(const map::Shot &curr_shot)
{
  constexpr unsigned int max_num_local_keyfrms{60};

  // count the number of sharing landmarks between the current frame and each of the neighbor keyframes
  // key: keyframe, value: number of sharing landmarks
  // std::unordered_map<KeyFrame*, unsigned int, KeyFrameCompare> keyfrm_weights;
  std::unordered_map<map::Shot *, unsigned int> keyfrm_weights;
  const auto &landmarks = curr_shot.GetLandmarks();
  const auto n_keypts = landmarks.size();
  for (unsigned int idx = 0; idx < n_keypts; ++idx)
  {
    auto lm = landmarks.at(idx);
    if (lm != nullptr)
    {
      // continue;
      // }
      // if (lm->will_be_erased()) {
      // std::cout << "lm->will_be_erased()" << std::endl;
      // exit(0);
      // kf.landmarks_.at(idx) = nullptr;
      //TODO: Write this maybe in a clean-up function!
      // frame.landmarks_.at(idx) = nullptr;
      // continue;
      // }

      const auto &observations = lm->GetObservations();
      for (const auto &obs : observations)
      {
        ++keyfrm_weights[obs.first];
      }
    }
  }

  if (keyfrm_weights.empty())
  {
    return std::vector<map::Shot *>();
  }

  // set the aforementioned keyframes as local keyframes
  // and find the nearest keyframe
  unsigned int max_weight = 0;
  // map::Shot* nearest_covisibility = nullptr;

  //ptr to Shot,
  // std::vector<map::Shot*> local_keyfrms;
  std::unordered_set<map::Shot *> local_keyfrms;
  // local_keyfrms.reserve(4 * keyfrm_weights.size());

  for (auto &keyfrm_weight : keyfrm_weights)
  {
    auto keyfrm = keyfrm_weight.first;
    const auto weight = keyfrm_weight.second;

    // if (keyfrm->will_be_erased()) {
    //     continue;
    // }

    // local_keyfrms.push_back(keyfrm);
    local_keyfrms.emplace(keyfrm);

    // avoid duplication
    // keyfrm->local_map_update_identifier = frame.frame_id;

    // update the nearest keyframe
    if (max_weight < weight)
    {
      max_weight = weight;
      // nearest_covisibility = keyfrm;
    }
  }
  std::cout << "local_keyfrms1: " << local_keyfrms.size() << std::endl;
  // add the second-order keyframes to the local landmarks
  auto add_local_keyframe = [&](map::Shot *keyfrm) {
    if (keyfrm == nullptr)
    {
      return false;
    }
    // if (keyfrm->will_be_erased()) {
    //     return false;
    // }
    // avoid duplication
    // if (keyfrm->local_map_update_identifier == frame.frame_id) {
    //     return false;
    // }
    // keyfrm->local_map_update_identifier = frame.frame_id;
    local_keyfrms.emplace(keyfrm);
    return true;
  };
  std::cout << "local_keyfrms2: " << local_keyfrms.size() << std::endl;
  const auto &n_local_keyfrms = local_keyfrms.size();
  for (auto iter = local_keyfrms.cbegin(), end = local_keyfrms.cend(); iter != end; ++iter)
  {
    if (max_num_local_keyfrms < n_local_keyfrms)
    {
      break;
    }

    auto keyfrm = *iter;

    // covisibilities of the neighbor keyframe
    const auto neighbors = keyfrm->slam_data_.graph_node_->get_top_n_covisibilities(10);
    for (auto neighbor : neighbors)
    {
      if (add_local_keyframe(neighbor))
      {
        break;
      }
    }

    // children of the spanning tree
    const auto spanning_children = keyfrm->slam_data_.graph_node_->get_spanning_children();
    for (auto child : spanning_children)
    {
      if (add_local_keyframe(child))
      {
        break;
      }
    }

    // parent of the spanning tree
    auto parent = keyfrm->slam_data_.graph_node_->get_spanning_parent();
    add_local_keyframe(parent);
  }
  std::cout << "local_keyfrms: " << local_keyfrms.size() << std::endl;
  return std::vector<map::Shot *>(local_keyfrms.begin(), local_keyfrms.end());
}

size_t
SlamUtilities::MatchShotToLocalMap(map::Shot &curr_shot, const slam::GuidedMatcher& matcher)
{

  constexpr unsigned int max_num_local_keyfrms{60};
  
  //First create a set of landmarks that don't need matching, i.e. the already seen ones
  std::unordered_set<map::Landmark*> matched_lms;
  for (const auto& lm : curr_shot.GetLandmarks())
  {
    if (lm != nullptr)
    {
      matched_lms.insert(lm);
      lm->slam_data_.IncreaseNumObservable();
    }
  }
  std::cout << "Created matched lms!" << std::endl;
  std::vector<std::pair<map::Landmark *, Eigen::Vector3d>,
              Eigen::aligned_allocator<Eigen::Vector3d>>
      local_landmarks; //Vector3f stores the reprojected point and its pred. scale
  // we get the local KFs!
  // e.g. KFs that see the same landmarks as the current frame
  // use a set to avoid duplicates
  std::unordered_set<map::Shot*> local_keyframes;
  Eigen::Vector2d reproj;
  size_t scale_level;
  // add the second-order keyframes to the local landmarks
  auto add_local_keyframe = [&](map::Shot *keyfrm) {
    if (keyfrm == nullptr)
    {
      return false;
    }
    const auto it = local_keyframes.emplace(keyfrm);
    if (!it.second) //try to add its landmarks
    {
      for (const auto& lm : keyfrm->GetLandmarks())
      {
        if (lm != nullptr && matched_lms.count(lm) == 0)
        {
          std::cout << "isobservable?" << std::endl;
          //try to reproject
          if (matcher.IsObservable(lm, *keyfrm, 0.5, reproj, scale_level))
          {
            std::cout << "observable?" << std::endl;
            local_landmarks.emplace_back(
              std::make_pair(lm, Eigen::Vector3d(reproj[0], reproj[1], scale_level)));
            std::cout << "added?" << std::endl;
          }
          //Don't try again
          matched_lms.insert(lm);
        }
      }
    }
    return it.second;
  };

  const auto &landmarks = curr_shot.GetLandmarks();
  const auto n_keypts = landmarks.size();
  for (unsigned int idx = 0; idx < n_keypts; ++idx)
  {
    const auto& lm = landmarks.at(idx);
    if (lm != nullptr)
    {
      std::cout << "Processing: " << lm->id_ << std::endl;
      for (const auto& obs : lm->GetObservations())
      {
        // local_keyframes.insert(obs.first);
        std::cout << "Adding " << obs.first << std::endl;
        add_local_keyframe(obs.first);
      }
    }
  }
  std::cout << "Created local lms and kfs!" << std::endl;

  if (local_keyframes.empty())
  {
    return 0;
  }


  // // add the second-order keyframes to the local landmarks
  // auto add_local_keyframe = [&](map::Shot *keyfrm) {
  //   if (keyfrm == nullptr)
  //   {
  //     return false;
  //   }
  //   const auto it = local_keyframes.emplace(keyfrm);
  //   return it.second;
  // };

  // Try to insert the max_num number of keyframes
  for (auto iter = local_keyframes.cbegin(), end = local_keyframes.cend();
       iter != end, max_num_local_keyfrms >= local_keyframes.size(); ++iter)
  {
    auto keyfrm = *iter;

    // covisibilities of the neighbor keyframe
    const auto neighbors = keyfrm->slam_data_.graph_node_->get_top_n_covisibilities(10);
    for (auto neighbor : neighbors)
    {
      if (add_local_keyframe(neighbor))
      {
        break;
      }
    }

    // children of the spanning tree
    const auto spanning_children = keyfrm->slam_data_.graph_node_->get_spanning_children();
    for (auto child : spanning_children)
    {
      if (add_local_keyframe(child))
      {
        break;
      }
    }

    // parent of the spanning tree
    auto parent = keyfrm->slam_data_.graph_node_->get_spanning_parent();
    add_local_keyframe(parent);
  }
  // std::cout << "local_keyfrms: " << local_keyframes.size() << std::endl;
  // return local_keyframes.size();
  // we get the local landmarks
  // Landmarks seen by the local KFs
  // std::unordered_set<map::Landmark*> local_landmarks;
  // for (auto keyframe : local_keyframes)
  // {
  //     for (auto lm : keyframe->GetLandmarks())
  //     {
  //         if (lm != nullptr)
  //         {
  //           local_landmarks.emplace(lm);
  //         }
  //     }
  // }

  // convert landmarks seen in current shot to unordered_set

  //make new vector
  // only add:
  // landmarks not in the set
  // and visible in the current frame

  //Assign landmarks to current frame
  return 0;
}

} // namespace slam