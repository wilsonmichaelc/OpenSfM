#pragma once
#include <Eigen/Eigen>
#include <map>
#include <opencv2/core.hpp>
namespace cslam
{
class KeyFrame;
class Landmark
{
public:
    Landmark(const size_t lm_id):lm_id_(lm_id){};

    void increase_num_observable(unsigned int num_observable = 1){ num_observable_ += num_observable; }
    void increase_num_observed(unsigned int num_observed = 1) { num_observed_ += num_observed; }
    float get_observed_ratio() const { return static_cast<float>(num_observed_)/num_observable_; }

    // Tracking information
    bool is_observable_in_tracking_; // true if can be reprojected to current frame
    size_t scale_level_in_tracking_;
    Eigen::Vector2f reproj_in_tracking_; // reprojected pixel position

    //! whether this landmark will be erased shortly or not
    bool will_be_erased() { return will_be_erased_; };
    cv::Mat get_descriptor() const { return descriptor_.clone(); }
    bool has_observation() const { return num_observations_ > 0; }
    size_t num_observations() const { return num_observations_; }
    void add_observation(KeyFrame*, const size_t idx);
private:
    const size_t lm_id_;
    size_t num_observations_ = 0;

    // track counter
    size_t num_observable_ = 1;
    size_t num_observed_ = 1;

    // ORB scale variances
    //! max valid distance between landmark and camera
    float min_valid_dist_ = 0;
    //! min valid distance between landmark and camera
    float max_valid_dist_ = 0;

        //! この3次元点を観測しているkeyframeについて，keyframe->lmのベクトルの平均値(規格化されてる)
    Eigen::Vector3f mean_normal_ = Eigen::Vector3f::Zero();

    //! representative descriptor
    cv::Mat descriptor_;

    //! reference keyframe
    KeyFrame* ref_keyfrm_;
    Eigen::Vector3f pos_w_;

    //! observations (keyframe and keypoint index)
    std::map<KeyFrame*, size_t> observations_;

    //! this landmark will be erased shortly or not
    bool will_be_erased_ = false; //Note: probably always false since we don't run in parallel




};
} // namespace cslam
