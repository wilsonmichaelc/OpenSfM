#pragma once
#include <vector>
namespace cslam
{
class Landmark;
class KeyFrame;
class LocalMapCleaner
{
public:
    LocalMapCleaner(){}
    void add_landmark(Landmark* new_lm)
    {
        fresh_landmarks_.push_back(new_lm);
    }
    void update_lms_after_kf_insert(KeyFrame* new_kf);
    size_t remove_redundant_landmarks(const size_t cur_keyfrm_id);
    //TODO: Implement
    void remove_redundant_keyframes(KeyFrame* new_kf);
    void count_redundant_observations(KeyFrame* keyfrm, unsigned int& num_valid_obs, unsigned int& num_redundant_obs) const;

private:
    std::vector<Landmark*> fresh_landmarks_;

};   
}