#include "local_map_cleaner.h"
#include "keyframe.h"
#include "landmark.h"
#include "camera.h"
#include "third_party/openvslam/util/guided_matching.h"
#include "slam_reconstruction.h"
namespace cslam
{

LocalMapCleaner::LocalMapCleaner(const GuidedMatcher& guided_matcher, SlamReconstruction* map_db): //, BrownPerspectiveCamera* camera):
    guided_matcher_(guided_matcher), camera_(guided_matcher.camera_), map_db_(map_db)
{

}

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

void
LocalMapCleaner::fuse_landmark_duplication(KeyFrame* curr_kf, const std::vector<KeyFrame*>& fuse_tgt_keyfrms) const
{
    auto cur_landmarks = curr_kf->landmarks_;
    std::cout << "Before first for" <<std::endl;

    // go through kfs and fuse!
    // reproject the landmarks observed in the current keyframe to each of the targets, and acquire
    // - additional matches
    // - duplication of matches
    // then, add matches and solve duplication
    for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            const auto n_fused = guided_matcher_.replace_duplication(fuse_tgt_keyfrm, cur_landmarks);
            std::cout << "Fused: " << n_fused << " for " << fuse_tgt_keyfrm->im_name_ << std::endl;
    }
    // std::cout << "After first for" <<std::endl;
    // reproject the landmarks observed in each of the targets to each of the current frame, and acquire
    // - additional matches
    // - duplication of matches
    // then, add matches and solve duplication
    std::unordered_set<Landmark*> candidate_landmarks_to_fuse;
    candidate_landmarks_to_fuse.reserve(fuse_tgt_keyfrms.size() * curr_kf->landmarks_.size());

    for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
        const auto fuse_tgt_landmarks = fuse_tgt_keyfrm->landmarks_;//fuse_tgt_keyfrm->get_landmarks();

        for (const auto lm : fuse_tgt_landmarks) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            if (static_cast<bool>(candidate_landmarks_to_fuse.count(lm))) {
                continue;
            }
            candidate_landmarks_to_fuse.insert(lm);
        }
    }
    std::cout << "After second for" <<std::endl;
    const auto n_fused = guided_matcher_.replace_duplication(curr_kf, candidate_landmarks_to_fuse);
    std::cout << "Fused: " << n_fused << " for curr" << curr_kf->im_name_ << std::endl;

}




void
LocalMapCleaner::update_new_keyframe(KeyFrame* curr_kf) const
{
    // update the geometries
    const auto& cur_landmarks = curr_kf->landmarks_;
    for (const auto lm : cur_landmarks) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        lm->compute_descriptor();
        lm->update_normal_and_depth();
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
            map_db_->erase_landmark(lm);
        }
        else {
            // hold decision because the state is NotClear
            iter++;
        }
    }
    std::cout << "remove_redundant_landmarks: " << num_removed << std::endl;
    return num_removed;
}
}