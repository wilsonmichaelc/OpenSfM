#include "local_map_cleaner.h"
#include "keyframe.h"
#include "landmark.h"
namespace cslam
{

void 
LocalMapCleaner::update_lms_after_kf_insert(KeyFrame* new_kf)
{
    // landmarks_
    for (unsigned int idx = 0; idx < new_kf->landmarks_.size(); ++idx) {
        auto lm = new_kf->landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // if `lm` does not have the observation information from `cur_keyfrm_`,
        // add the association between the keyframe and the landmark
        if (lm->is_observed_in_keyframe(new_kf)) {
            // if `lm` is correctly observed, make it be checked by the local map cleaner
            // local_map_cleaner_->add_fresh_landmark(lm);
            fresh_landmarks_.push_back(lm);
            continue;
        }

        // update connection
        lm->add_observation(new_kf, idx);
        // update geometry
        lm->update_normal_and_depth();
        lm->compute_descriptor();
    }
}

size_t
LocalMapCleaner::remove_redundant_landmarks(const size_t cur_keyfrm_id) {
    constexpr float observed_ratio_thr = 0.3;
    constexpr unsigned int num_reliable_keyfrms = 2;
    // const unsigned int num_obs_thr = is_monocular_ ? 2 : 3;
    constexpr size_t num_obs_thr{2};

    // states of observed landmarks
    enum class lm_state_t { Valid, Invalid, NotClear };

    unsigned int num_removed = 0;
    auto iter = fresh_landmarks_.begin();
    while (iter != fresh_landmarks_.end()) {
        auto lm = *iter;

        // decide the state of lms the buffer
        auto lm_state = lm_state_t::NotClear;
        if (lm->will_be_erased()) {
            // in case `lm` will be erased
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }
        else if (lm->get_observed_ratio() < observed_ratio_thr) {
            // if `lm` is not reliable
            // remove `lm` from the buffer and the database
            lm_state = lm_state_t::Invalid;
        }
        else if (num_reliable_keyfrms + lm->ref_kf_id_ <= cur_keyfrm_id
                 && lm->num_observations() <= num_obs_thr) {
            // if the number of the observers of `lm` is small after some keyframes were inserted
            // remove `lm` from the buffer and the database
            lm_state = lm_state_t::Invalid;
        }
        else if (num_reliable_keyfrms + 1U + lm->ref_kf_id_ <= cur_keyfrm_id) {
            // if the number of the observers of `lm` is sufficient after some keyframes were inserted
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }

        // select to remove `lm` according to the state
        if (lm_state == lm_state_t::Valid) {
            iter = fresh_landmarks_.erase(iter);
        }
        else if (lm_state == lm_state_t::Invalid) {
            ++num_removed;
            lm->prepare_for_erasing();
            iter = fresh_landmarks_.erase(iter);
        }
        else {
            // hold decision because the state is NotClear
            iter++;
        }
    }

    return num_removed;
}


}