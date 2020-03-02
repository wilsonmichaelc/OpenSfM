#pragma once
#include <unordered_set>
#include <vector>


namespace cslam
{
class KeyFrame;
class Frame;
class Landmark;
/**
 * 
 * 
 * Contains static SLAM util methods
 * 
 * 
 * 
 * */
class SlamUtilities
{
public:
    // static std::unordered_set<KeyFrame*>
    static std::vector<KeyFrame*>
    get_second_order_covisibilities_for_kf(const KeyFrame* kf, const size_t first_order_thr, const size_t second_order_thr);
    
    static void
    update_new_keyframe(KeyFrame& curr_kf);
    static std::vector<KeyFrame*> 
    update_local_keyframes(Frame& frame);
    static std::vector<cslam::Landmark*>
    update_local_landmarks(const std::vector<cslam::KeyFrame*>& local_keyframes, const size_t curr_frm_id);
};
}