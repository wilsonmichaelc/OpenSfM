#pragma once
#include <vector>
#include <unordered_set>

namespace cslam
{
class Landmark;
class KeyFrame;
class BrownPerspectiveCamera;
class GuidedMatcher;
class LocalMapCleaner
{
public:
    LocalMapCleaner() = delete;
    LocalMapCleaner(const GuidedMatcher& guided_matcher); //, BrownPerspectiveCamera* camera);
    void add_landmark(Landmark* new_lm)
    {
        fresh_landmarks_.push_back(new_lm);
    }
    void update_lms_after_kf_insert(KeyFrame* new_kf);
    size_t remove_redundant_landmarks(const size_t cur_keyfrm_id);
    //TODO: Implement
    void remove_redundant_keyframes(KeyFrame* new_kf);
    void count_redundant_observations(KeyFrame* keyfrm, unsigned int& num_valid_obs, unsigned int& num_redundant_obs) const;
    void update_new_keyframe(KeyFrame* curr_kf) const;
    // void fuse_landmark_duplication(KeyFrame* curr_kf, const std::unordered_set<KeyFrame*>& fuse_tgt_keyfrms) const;
    void fuse_landmark_duplication(KeyFrame* curr_kf, const std::vector<KeyFrame*>& fuse_tgt_keyfrms) const;
private:
    std::vector<Landmark*> fresh_landmarks_;
    const GuidedMatcher& guided_matcher_;
    const BrownPerspectiveCamera& camera_;

};   
}