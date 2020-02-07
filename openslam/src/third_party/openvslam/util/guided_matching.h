#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include "third_party/openvslam/feature/orb_params.h"
namespace cslam{ 
    class Frame;
    class Landmark;
    class KeyFrame;
    class BrownPerspectiveCamera;
}
namespace cslam
{
struct GridParameters
{
    GridParameters(unsigned int grid_col_, unsigned int grid_rows,
                   float img_min_width, float img_min_height,
                   float img_max_width, float img_max_height,
                   float inv_cell_width, float inv_cell_height);
    unsigned int grid_cols_, grid_rows_;
    float img_min_width_, img_min_height_;
    float img_max_width_, img_max_height_;
    float inv_cell_width_, inv_cell_height_;

    bool in_grid(const Eigen::Vector2f& pt2D) const { return in_grid(pt2D[0], pt2D[1]); }
    bool in_grid(const float x, const float y) const
    {
        return img_min_width_ < x && img_max_width_ > x && img_min_height_ < y && img_max_height_ > y;  
    }
};

using CellIndices = std::vector<std::vector<std::vector<unsigned int>>>;
using MatchIndices = std::vector<std::pair<size_t, size_t>>;
using OrbFeature = Eigen::Matrix<float, 1, 5>;
class GuidedMatcher
{
public:
    static constexpr unsigned int HAMMING_DIST_THR_LOW = 50;
    static constexpr unsigned int HAMMING_DIST_THR_HIGH = 100;
    // static constexpr unsigned int HAMMING_DIST_THR_LOW = 80;
    // static constexpr unsigned int HAMMING_DIST_THR_HIGH = 120;
    static constexpr unsigned int MAX_HAMMING_DIST = 256;

    //! ORB特徴量間のハミング距離を計算する
    static inline unsigned int 
    compute_descriptor_distance_32(const cv::Mat& desc_1, const cv::Mat& desc_2);
    GuidedMatcher(const GridParameters& grid_params, const BrownPerspectiveCamera& camera);
    const GridParameters& grid_params_;
    const BrownPerspectiveCamera& camera_;
    void assign_points_to_grid(const Eigen::MatrixXf& undist_keypts, CellIndices& keypt_indices_in_cells);
    CellIndices assign_keypoints_to_grid(const Eigen::MatrixXf& undist_keypts);

    void distribute_keypoints_to_grid_frame(cslam::Frame& frame);

    void distribute_keypoints_to_grid(const std::vector<cv::KeyPoint>& undist_keypts,
                                    CellIndices& keypt_indices_in_cells);

    // std::vector<size_t> 
    // get_keypoints_in_cell(const Eigen::MatrixXf& undist_keypts,
    //                     const CellIndices& keypt_indices_in_cells,
    //                     const float ref_x, const float ref_y, const float margin,
    //                     const int min_level, const int max_level);

    std::vector<size_t> 
    get_keypoints_in_cell(const std::vector<cv::KeyPoint>& undist_keypts,
                          const CellIndices& keypt_indices_in_cells,
                          const float ref_x, const float ref_y, const float margin,
                          const int min_level, const int max_level) const;

    // MatchIndices 
    // match_frame_to_frame_py(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
    //                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
    //                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
    //                     const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
    //                     const size_t margin);

    // std::vector<size_t> 
    // match_frame_to_frame_dbg(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
    //                     //  const Eigen::MatrixXi& desc_1, const Eigen::MatrixXi& desc_2, 
    //                     //  const Eigen::MatrixX& desc_1, const Eigen::MatrixXf& desc_2, 
    //                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
    //                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
    //                     const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
    //                     const size_t margin);
    MatchIndices 
    match_frame_to_frame(const cslam::Frame& frame1, const cslam::Frame& frame2,
                        const Eigen::MatrixX2f& prevMatched,
                        const size_t margin);


    // TODO: Think about the margin. Maybe make it dynamic depending on the depth of the feature!!
    size_t
    match_frame_and_landmarks(const std::vector<float>& scale_factors, //const openvslam::feature::orb_params& orb_params,
                            cslam::Frame& frame, std::vector<cslam::Landmark*>& local_landmarks, const float margin);
    std::vector<cslam::Landmark*>
    update_local_landmarks(const std::vector<cslam::KeyFrame*>& local_keyframes, const size_t curr_frm_id);

    size_t
    match_current_and_last_frame(cslam::Frame& curr_frm, const cslam::Frame& last_frm, const float margin);
};

// Eigen::Matrix4f
// track_to_last_frame(const GridParameters& grid_params, const Eigen::Matrix4f& T_wc_init,
//                     const cslam::Frame& last_frame, cslam::Frame& curr_frame);
// {
//     Eigen::Matrix4f T_wc = T_wc_ini
// }
};