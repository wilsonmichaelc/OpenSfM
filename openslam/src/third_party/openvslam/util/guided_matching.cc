#include "guided_matching.h"
#include "angle_checker.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <unordered_set>
#include "slam_datastructures/frame.h"
#include "slam_datastructures/landmark.h"
#include "slam_datastructures/keyframe.h"
#include "slam_datastructures/camera.h"

namespace cslam
{

GridParameters::GridParameters(unsigned int grid_cols, unsigned int grid_rows,
                float img_min_width, float img_min_height,
                float img_max_width, float img_max_height,
                float inv_cell_width, float inv_cell_height):
                grid_cols_(grid_cols), grid_rows_(grid_rows),
                img_min_width_(img_min_width), img_min_height_(img_min_height),
                img_max_width_(img_max_width), img_max_height_(img_max_height),
                inv_cell_width_(inv_cell_width), inv_cell_height_(inv_cell_height)
                {}


GuidedMatcher::GuidedMatcher(const GridParameters& grid_params, const BrownPerspectiveCamera& camera):
    grid_params_(grid_params), camera_(camera)
{

}


void
GuidedMatcher::assign_points_to_grid(const Eigen::MatrixXf& undist_keypts, CellIndices& keypt_indices_in_cells)
{
    const size_t num_pts = undist_keypts.rows();
    const size_t num_to_reserve = 0.5 * num_pts / (grid_params_.grid_cols_*grid_params_.grid_rows_);
    keypt_indices_in_cells.resize(grid_params_.grid_cols_);
    for (auto& keypt_indices_in_row : keypt_indices_in_cells) {
        keypt_indices_in_row.resize(grid_params_.grid_rows_);
        for (auto& keypt_indices_in_cell : keypt_indices_in_row) {
            keypt_indices_in_cell.reserve(num_to_reserve);
        }
    }
    for (size_t idx = 0; idx < num_pts; ++idx) {
        // const auto& keypt = undist_keypts.at(idx);
        const Eigen::Vector2f pt = undist_keypts.block<1,2>(idx,0);
        // std::cout << "pt: " << pt.transpose() << std::endl;
        const int cell_idx_x = std::round((pt[0] - grid_params_.img_min_width_) * grid_params_.inv_cell_width_);
        const int cell_idx_y = std::round((pt[1] - grid_params_.img_min_height_) * grid_params_.inv_cell_height_);
        if ((0 <= cell_idx_x && cell_idx_x < static_cast<int>(grid_params_.grid_cols_)
            && 0 <= cell_idx_y && cell_idx_y < static_cast<int>(grid_params_.grid_rows_)))
        {
            keypt_indices_in_cells.at(cell_idx_x).at(cell_idx_y).push_back(idx);
        }
    }
}
CellIndices 
GuidedMatcher::assign_keypoints_to_grid(const Eigen::MatrixXf& undist_keypts) {
    CellIndices keypt_indices_in_cells;
    assign_points_to_grid(undist_keypts, keypt_indices_in_cells);
    return keypt_indices_in_cells;
}

void
GuidedMatcher::distribute_keypoints_to_grid_frame(cslam::Frame& frame)
{
    distribute_keypoints_to_grid(frame.undist_keypts_, frame.keypts_indices_in_cells_);
}

void 
GuidedMatcher::distribute_keypoints_to_grid(const std::vector<cv::KeyPoint>& undist_keypts, CellIndices& keypt_indices_in_cells)
{
    const size_t num_pts = undist_keypts.size();
    const size_t num_to_reserve = 0.5 * num_pts / (grid_params_.grid_cols_*grid_params_.grid_rows_);
    keypt_indices_in_cells.resize(grid_params_.grid_cols_);
    for (auto& keypt_indices_in_row : keypt_indices_in_cells) {
        keypt_indices_in_row.resize(grid_params_.grid_rows_);
        for (auto& keypt_indices_in_cell : keypt_indices_in_row) {
            keypt_indices_in_cell.reserve(num_to_reserve);
        }
    }
    for (size_t idx = 0; idx < num_pts; ++idx) {
        const auto& keypt = undist_keypts.at(idx);
        // const Eigen::Vector2f pt = undist_keypts.block<1,2>(idx,0);
        // std::cout << "pt: " << pt.transpose() << std::endl;
        const int cell_idx_x = std::round((keypt.pt.x - grid_params_.img_min_width_) * grid_params_.inv_cell_width_);
        const int cell_idx_y = std::round((keypt.pt.y - grid_params_.img_min_height_) * grid_params_.inv_cell_height_);
        if ((0 <= cell_idx_x && cell_idx_x < static_cast<int>(grid_params_.grid_cols_)
            && 0 <= cell_idx_y && cell_idx_y < static_cast<int>(grid_params_.grid_rows_)))
        {
            keypt_indices_in_cells.at(cell_idx_x).at(cell_idx_y).push_back(idx);
        }
    }
}
// MatchIndices 
// match_frame_to_frame(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
//                     //  const Eigen::MatrixXi& desc_1, const Eigen::MatrixXi& desc_2, 
//                     //  const Eigen::MatrixX& desc_1, const Eigen::MatrixXf& desc_2, 
//                      const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
//                      const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2) //,
//                      const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
//                      const GridParameters& grid_params, const size_t margin);
// {
//     MatchIndices matches;
//     std::cout << "udnist_keypts_1,2" << std::endl;
//     return matches;
// }

//! ORB特徴量間のハミング距離を計算する
// inline unsigned int 
// GuidedMatcher::compute_descriptor_distance_32(const cv::Mat& desc_1, const cv::Mat& desc_2)
// {
//     // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

//     constexpr uint32_t mask_1 = 0x55555555U;
//     constexpr uint32_t mask_2 = 0x33333333U;
//     constexpr uint32_t mask_3 = 0x0F0F0F0FU;
//     constexpr uint32_t mask_4 = 0x01010101U;

//     const auto pa = desc_1.ptr<uint32_t>();
//     const auto pb = desc_2.ptr<uint32_t>();

//     unsigned int dist = 0;

//     for (unsigned int i = 0; i < 8; ++i, ++pa, ++pb) {
//         auto v = *pa ^*pb;
//         v -= ((v >> 1) & mask_1);
//         v = (v & mask_2) + ((v >> 2) & mask_2);
//         dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 24;
//     }

//     return dist;
// }




/**
 * 
 * unidst_keypts should have x,y,...
 *      keys.at<float>(i, 0) = kpts[i].pt.x;
        keys.at<float>(i, 1) = kpts[i].pt.y;
        keys.at<float>(i, 2) = kpts[i].size;
        keys.at<float>(i, 3) = kpts[i].angle;
        keys.at<float>(i, 4) = kpts[i].octave;
 * 
 * prev_matched are the updated "matched" coordinates of 1
 */
// MatchIndices
// GuidedMatcher::match_frame_to_frame_py(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
//                      Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
//                      Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
//                      const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
//                      const size_t margin)
// {
//     constexpr auto check_orientation_{true};
//     constexpr float lowe_ratio_{0.9};
//     // std::cout << "grid: " << grid_params.img_min_width << "/" << grid_params.img_min_height << " inv: " << grid_params.inv_cell_height << "," << grid_params.inv_cell_width
//     //                       << grid_params.grid_cols << "/" << grid_params.grid_rows << std::endl; 
//     // std::cout << "match_frame_to_frame" << std::endl;
//     const size_t num_pts_1 = undist_keypts_1.rows();
//     const size_t num_pts_2 = undist_keypts_2.rows();
//     // std::cout << "match_frame_to_frame" << num_pts_1 << "/" << num_pts_2  
//     //           << " desc: " << desc_1.rows() << ", " << desc_1.cols()  
//     //           << " desc2: " << desc_2.rows() << ", " << desc_2.cols() << std::endl;
//     MatchIndices matches; // Index in 1, Index in 2
//     matches.reserve(num_pts_1);
//     std::vector<unsigned int> matched_dists_in_frm_2(num_pts_2, MAX_HAMMING_DIST);
//     std::vector<int> matched_indices_1_in_frm_2(num_pts_2, -1);
//     std::vector<int> matched_indices_2_in_frm_1 = std::vector<int>(num_pts_1, -1);
//     size_t num_matches = 0; // Todo: should be the same as matches.size()
//     openvslam::match::angle_checker<int> angle_checker;
//     // Wrap the descriptors in a CV Mat to make handling easier
//     cv::Mat desc1_cv;
//     cv::eigen2cv(desc_1, desc1_cv);
//     cv::Mat desc2_cv;
//     cv::eigen2cv(desc_2, desc2_cv);
//     for (size_t idx_1 = 0; idx_1 < num_pts_1; ++idx_1)
//     {
//         // f1 = x, y, size, angle, octave
//         const OrbFeature f1 = undist_keypts_1.block<1,5>(idx_1,0);
//         const float scale_1 = f1[4];
//         // std::cout << "f1: " << f1.transpose() << std::endl;
//         if (scale_1 < 0) continue;
//         // Now, 
//         const auto indices = get_keypoints_in_cell(undist_keypts_2, cell_indices_2, f1[0], f1[1], margin, scale_1, scale_1);
//         // std::cout << "indices: " << indices.size() << std::endl;
//         if (indices.empty()) continue; // No valid match
//         // std::cout << "indices: " << indices.size() << std::endl;

//         // Read the descriptor
//         const auto& d1 = desc1_cv.row(idx_1);
//         auto best_hamm_dist = MAX_HAMMING_DIST;
//         auto second_best_hamm_dist = MAX_HAMMING_DIST;
//         int best_idx_2 = -1;
//         for (const auto idx_2 : indices) 
//         {
//             const auto& d2 = desc2_cv.row(idx_2);
//             const auto hamm_dist = compute_descriptor_distance_32(d1, d2);
//             // std::cout << "d1: " << d1 << "\n d2: " << d2 << "=" << hamm_dist << std::endl;
//             // through if the point already matched is closer
//             if (matched_dists_in_frm_2.at(idx_2) <= hamm_dist) {
//                 // std::cout << "cont here1" << std::endl;
//                 continue;
//             }
//             if (hamm_dist < best_hamm_dist) {
//                 second_best_hamm_dist = best_hamm_dist;
//                 best_hamm_dist = hamm_dist;
//                 best_idx_2 = idx_2;
//             }
//             else if (hamm_dist < second_best_hamm_dist) {
//                 second_best_hamm_dist = hamm_dist;
//             }

//         }

//         if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
//             // std::cout << "cont HAMMING_DIST_THR_LOW" << std::endl;
//             continue;
//         }

//         // ratio test
//         if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist)) {
//             // std::cout << "cont lowe_ratio_" << std::endl;

//             continue;
//         }

//         const auto prev_idx_1 = matched_indices_1_in_frm_2.at(best_idx_2);
//         if (0 <= prev_idx_1) {
//             matched_indices_2_in_frm_1.at(prev_idx_1) = -1;
//             --num_matches;
//         }

//         // 互いの対応情報を記録する
//         matched_indices_2_in_frm_1.at(idx_1) = best_idx_2;
//         matched_indices_1_in_frm_2.at(best_idx_2) = idx_1;
//         matched_dists_in_frm_2.at(best_idx_2) = best_hamm_dist;
//         ++num_matches;
//         // std::cout << "num_matches: " << num_matches << std::endl;



//         if (check_orientation_) {
//             // const auto delta_angle
//                     // = undist_keypts_1.at(idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
//             const auto delta_angle
//                     = undist_keypts_1(idx_1, 3) -  undist_keypts_2(best_idx_2, 3); // (idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
//             angle_checker.append_delta_angle(delta_angle, idx_1);
//         }
//     }

//     if (check_orientation_) {
//         const auto invalid_matches = angle_checker.get_invalid_matches();
//         for (const auto invalid_idx_1 : invalid_matches) {
//             if (0 <= matched_indices_2_in_frm_1.at(invalid_idx_1)) {
//                 matched_indices_2_in_frm_1.at(invalid_idx_1) = -1;
//                 --num_matches;
//             }
//         }
//     }

    
//     for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
//         const auto idx_2 = matched_indices_2_in_frm_1.at(idx_1);
//         if (idx_2 >= 0)
//         {
//             matches.emplace_back(std::make_pair(idx_1, idx_2));
//             std::cout << "Found match at: " << idx_1 << "/" << idx_2 << std::endl;
//         }
//     } 

//     // TODO: update this out of the loop!
//     // previous matchesを更新する
//     // for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
//     //     if (0 <= matched_indices_2_in_frm_1.at(idx_1)) {
//     //         prev_matched_pts.at(idx_1) = undist_keypts_2.at(matched_indices_2_in_frm_1.at(idx_1)).pt;
//     //     }
//     // }



//     return matches;
// };

MatchIndices
GuidedMatcher::match_frame_to_frame(const cslam::Frame& frame1, const cslam::Frame& frame2,
                     const Eigen::MatrixX2f& prevMatched, const size_t margin)
{
    constexpr auto check_orientation_{true};
    constexpr float lowe_ratio_{0.9};
    const size_t num_pts_1 = frame1.undist_keypts_.size();
    const size_t num_pts_2 = frame2.undist_keypts_.size();
    MatchIndices matches; // Index in 1, Index in 2
    matches.reserve(num_pts_1);
    std::vector<unsigned int> matched_dists_in_frm_2(num_pts_2, MAX_HAMMING_DIST);
    std::vector<int> matched_indices_1_in_frm_2(num_pts_2, -1);
    std::vector<int> matched_indices_2_in_frm_1 = std::vector<int>(num_pts_1, -1);
    size_t num_matches = 0; // Todo: should be the same as matches.size()
    openvslam::match::angle_checker<int> angle_checker;
    for (size_t idx_1 = 0; idx_1 < num_pts_1; ++idx_1)
    {
        // f1 = x, y, size, angle, octave
        const auto& u_kpt_1 = frame1.undist_keypts_.at(idx_1);
        const float scale_1 = u_kpt_1.octave;
        if (scale_1 < 0) continue;
        const auto indices = get_keypoints_in_cell(frame2.undist_keypts_, frame2.keypts_indices_in_cells_, 
                                                   u_kpt_1.pt.x, u_kpt_1.pt.y, margin, scale_1, scale_1);
        if (indices.empty()) continue; // No valid match

        // Read the descriptor
        const auto& d1 = frame1.descriptors_.row(idx_1);
        auto best_hamm_dist = MAX_HAMMING_DIST;
        auto second_best_hamm_dist = MAX_HAMMING_DIST;
        int best_idx_2 = -1;
        for (const auto idx_2 : indices) 
        {
            const auto& d2 = frame2.descriptors_.row(idx_2);
            const auto hamm_dist = compute_descriptor_distance_32(d1, d2);
            // through if the point already matched is closer
            if (matched_dists_in_frm_2.at(idx_2) <= hamm_dist) {
                continue;
            }
            if (hamm_dist < best_hamm_dist) {
                second_best_hamm_dist = best_hamm_dist;
                best_hamm_dist = hamm_dist;
                best_idx_2 = idx_2;
            }
            else if (hamm_dist < second_best_hamm_dist) {
                second_best_hamm_dist = hamm_dist;
            }

        }

        if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
            continue;
        }

        // ratio test
        if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist)) {
            continue;
        }

        const auto prev_idx_1 = matched_indices_1_in_frm_2.at(best_idx_2);
        if (0 <= prev_idx_1) {
            matched_indices_2_in_frm_1.at(prev_idx_1) = -1;
            --num_matches;
        }

        // 互いの対応情報を記録する
        matched_indices_2_in_frm_1.at(idx_1) = best_idx_2;
        matched_indices_1_in_frm_2.at(best_idx_2) = idx_1;
        matched_dists_in_frm_2.at(best_idx_2) = best_hamm_dist;
        ++num_matches;

        if (check_orientation_) {
            const auto delta_angle
                    = frame1.undist_keypts_.at(idx_1).angle - frame2.undist_keypts_.at(best_idx_2).angle;
            angle_checker.append_delta_angle(delta_angle, idx_1);
        }
    }

    if (check_orientation_) {
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx_1 : invalid_matches) {
            if (0 <= matched_indices_2_in_frm_1.at(invalid_idx_1)) {
                matched_indices_2_in_frm_1.at(invalid_idx_1) = -1;
                --num_matches;
            }
        }
    }

    
    for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
        const auto idx_2 = matched_indices_2_in_frm_1.at(idx_1);
        if (idx_2 >= 0)
        {
            matches.emplace_back(std::make_pair(idx_1, idx_2));
            // std::cout << "Found match at: " << idx_1 << "/" << idx_2 << std::endl;
        }
    } 

    // TODO: update this out of the loop!
    // previous matchesを更新する
    // for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
    //     if (0 <= matched_indices_2_in_frm_1.at(idx_1)) {
    //         prev_matched_pts.at(idx_1) = undist_keypts_2.at(matched_indices_2_in_frm_1.at(idx_1)).pt;
    //     }
    // }
    return matches;
};

// std::vector<size_t>
// match_frame_to_frame_dbg(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
//                      Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
//                      Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
//                      const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
//                      const GridParameters& grid_params, const size_t margin)
// {
//     constexpr auto check_orientation_{true};
//     constexpr float lowe_ratio_{0.9};
//     std::cout << "grid: " << grid_params.img_min_width << "/" << grid_params.img_min_height << " inv: " << grid_params.inv_cell_height << "," << grid_params.inv_cell_width
//                           << grid_params.grid_cols << "/" << grid_params.grid_rows << std::endl; 
//     std::cout << "match_frame_to_frame" << std::endl;
//     const size_t num_pts_1 = undist_keypts_1.rows();
//     const size_t num_pts_2 = undist_keypts_2.rows();
//     std::cout << "match_frame_to_frame" << num_pts_1 << "/" << num_pts_2  
//               << " desc: " << desc_1.rows() << ", " << desc_1.cols()  
//               << " desc2: " << desc_2.rows() << ", " << desc_2.cols() << std::endl;
//     MatchIndices matches; // Index in 1, Index in 2
//     matches.reserve(num_pts_1);
//     std::vector<unsigned int> matched_dists_in_frm_2(num_pts_2, MAX_HAMMING_DIST);
//     std::vector<int> matched_indices_1_in_frm_2(num_pts_2, -1);
//     std::vector<int> matched_indices_2_in_frm_1 = std::vector<int>(num_pts_1, -1);
//     size_t num_matches = 0; // Todo: should be the same as matches.size()
//     openvslam::match::angle_checker<int> angle_checker;
//     // Wrap the descriptors in a CV Mat to make handling easier
//     cv::Mat desc1_cv(desc_1.rows(), desc_1.cols(), CV_8UC1, desc_1.data());
//     cv::Mat desc2_cv(desc_2.rows(), desc_2.cols(), CV_8UC1, desc_2.data());
    
//     for (size_t idx_1 = 0; idx_1 < num_pts_1; ++idx_1)
//     {
//         // f1 = x, y, size, angle, octave
//         const OrbFeature f1 = undist_keypts_1.block<1,5>(idx_1,0);
//         // std::cout << "idx_1: " << idx_1 << " f1: " << f1 << std::endl;
//         const float scale_1 = f1[4];
//         // std::cout << "f1: " << f1.transpose() << std::endl;
//         if (scale_1 < 0) continue;
//         // Now, 
//         const auto indices = get_keypoints_in_cell(grid_params, undist_keypts_2, cell_indices_2, f1[0], f1[1], margin, scale_1, scale_1);
//         std::cout << "indices: " << indices.size() << ", " << f1[0] << ", " << f1[1] << std::endl;
//         if (indices.empty()) continue; // No valid match
//         std::cout << "indices: " << indices.size() << "idx: " << idx_1 
//                   << " f1: " << f1 << std::endl;
//         for (const auto idx_2 : indices) 
//         {
//             std::cout << "idx_2: " << idx_2 <<": "
//                       << undist_keypts_2.block<1,5>(idx_2,0) << std::endl;
//         }
//         // return indices;
//         // Read the descriptor
//         const auto& d1 = desc1_cv.row(idx_1);
//         auto best_hamm_dist = MAX_HAMMING_DIST;
//         auto second_best_hamm_dist = MAX_HAMMING_DIST;
//         int best_idx_2 = -1;
//         for (const auto idx_2 : indices) 
//         {
//             const auto& d2 = desc2_cv.row(idx_2);
//             const auto hamm_dist = compute_descriptor_distance_32(d1, d2);
//             std::cout << "d1: " << d1 << "\n d2: " << d2 << "=" << hamm_dist << std::endl;
//             // through if the point already matched is closer
//             if (matched_dists_in_frm_2.at(idx_2) <= hamm_dist) {
//                 std::cout << "cont here1" << std::endl;
//                 continue;
//             }
//             if (hamm_dist < best_hamm_dist) {
//                 second_best_hamm_dist = best_hamm_dist;
//                 best_hamm_dist = hamm_dist;
//                 best_idx_2 = idx_2;
//             }
//             else if (hamm_dist < second_best_hamm_dist) {
//                 second_best_hamm_dist = hamm_dist;
//             }

//         }

//         if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
//             std::cout << "cont HAMMING_DIST_THR_LOW" << std::endl;
//             continue;
//         }

//         // ratio test
//         if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist)) {
//             std::cout << "cont lowe_ratio_" << std::endl;

//             continue;
//         }

//         const auto prev_idx_1 = matched_indices_1_in_frm_2.at(best_idx_2);
//         if (0 <= prev_idx_1) {
//             matched_indices_2_in_frm_1.at(prev_idx_1) = -1;
//             --num_matches;
//         }

//         // 互いの対応情報を記録する
//         matched_indices_2_in_frm_1.at(idx_1) = best_idx_2;
//         matched_indices_1_in_frm_2.at(best_idx_2) = idx_1;
//         matched_dists_in_frm_2.at(best_idx_2) = best_hamm_dist;
//         ++num_matches;
//         std::cout << "num_matches: " << num_matches << std::endl;



//         if (check_orientation_) {
//             // const auto delta_angle
//                     // = undist_keypts_1.at(idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
//             const auto delta_angle
//                     = undist_keypts_1(idx_1, 3) -  undist_keypts_2(best_idx_2, 3); // (idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
//             angle_checker.append_delta_angle(delta_angle, idx_1);
//         }
//         return indices;
//     }

//     if (check_orientation_) {
//         const auto invalid_matches = angle_checker.get_invalid_matches();
//         for (const auto invalid_idx_1 : invalid_matches) {
//             if (0 <= matched_indices_2_in_frm_1.at(invalid_idx_1)) {
//                 matched_indices_2_in_frm_1.at(invalid_idx_1) = -1;
//                 --num_matches;
//             }
//         }
//     }

    
//     for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
//         const auto idx_2 = matched_indices_2_in_frm_1.at(idx_1);
//         if (idx_2 >= 0)
//         {
//             matches.emplace_back(std::make_pair(idx_1, idx_2));
//             std::cout << "Found match at: " << idx_1 << "/" << idx_2 << std::endl;
//         }
//     } 

//     // TODO: update this out of the loop!
//     // previous matchesを更新する
//     // for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
//     //     if (0 <= matched_indices_2_in_frm_1.at(idx_1)) {
//     //         prev_matched_pts.at(idx_1) = undist_keypts_2.at(matched_indices_2_in_frm_1.at(idx_1)).pt;
//     //     }
//     // }


//     std::vector<size_t> dummy;
//     return dummy;
// };






// void match_points_to_frame(){std::cout << "match_points_to_frame" << std::endl;};

// std::vector<size_t> 
// get_keypoints_in_cell(const GridParameters& grid_params, const Eigen::MatrixXf& undist_keypts,
//                       const CellIndices& keypt_indices_in_cells,
//                       const float ref_x, const float ref_y, const float margin,
//                       const int min_level, const int max_level)
// {
//     std::vector<size_t> indices;
//     indices.reserve(undist_keypts.size());
//     const int min_cell_idx_x = std::max(0, cvFloor((ref_x - grid_params.img_min_width_ - margin) * grid_params.inv_cell_width_));

//     if (static_cast<int>(grid_params.grid_cols) <= min_cell_idx_x) {
//         return indices;
//     }

//     const int max_cell_idx_x = std::min(static_cast<int>(grid_params.grid_cols_ - 1), cvCeil((ref_x - grid_params.img_min_width_ + margin) * grid_params.inv_cell_width_));
//     // std::cout << "max_cell_idx_x: " << max_cell_idx_x << std::endl;
//     if (max_cell_idx_x < 0) {
//         return indices;
//     }

//     const int min_cell_idx_y = std::max(0, cvFloor((ref_y - grid_params.img_min_height_ - margin) * grid_params.inv_cell_height_));
//     // std::cout << "min_cell_idx_y: " << min_cell_idx_y << std::endl;
//     if (static_cast<int>(grid_params.grid_rows_) <= min_cell_idx_y) {
//         return indices;
//     }

//     const int max_cell_idx_y = std::min(static_cast<int>(grid_params.grid_rows_- 1), cvCeil((ref_y - grid_params.img_min_height_ + margin) * grid_params.inv_cell_height_));
//     // std::cout << "max_cell_idx_y: " << max_cell_idx_y << std::endl;
//     if (max_cell_idx_y < 0) {
//         return indices;
//     }

//     const bool check_level = (0 < min_level) || (0 <= max_level);
//     std::cout << "check_level: " << check_level << std::endl;
//     for (int cell_idx_x = min_cell_idx_x; cell_idx_x <= max_cell_idx_x; ++cell_idx_x) {
//         for (int cell_idx_y = min_cell_idx_y; cell_idx_y <= max_cell_idx_y; ++cell_idx_y) {
//             const auto& keypt_indices_in_cell = keypt_indices_in_cells.at(cell_idx_x).at(cell_idx_y);
//             // std::cout << "keypt_indices_in_cell: " << keypt_indices_in_cell.size() << std::endl;

//             if (keypt_indices_in_cell.empty()) {
//                 continue;
//             }

//             for (unsigned int idx : keypt_indices_in_cell) {
//                 const OrbFeature feature = undist_keypts.block<1,5>(idx,0);
//                 // std::cout << " feature: " << feature << " ref: " << ref_x << "/" << ref_y << std::endl;
//                 const float octave = feature[4];
//                 if (check_level) {
//                     if (octave < min_level || (0 <= max_level && max_level < octave)) {
//                         // std::cout << "cont lvl not matching!" << "min_level: " << min_level << " max_level: " 
//                         //   << max_level << " octave: " << octave << std::endl;
//                         continue;
//                     }
//                 }

//                 const float dist_x = feature[0] - ref_x;
//                 const float dist_y = feature[1] - ref_y;
//                 // std::cout << "dist: " << dist_x << "/" << dist_y << " ref: " << ref_x << "/" << ref_y << std::endl;
//                 if (std::abs(dist_x) < margin && std::abs(dist_y) < margin) {
//                     indices.push_back(idx);
//                     // std::cout << "idx: " << idx << std::endl;
//                     // exit(0);
//                 }
//             }
//         }
//     }

//     return indices;
// }

std::vector<size_t> 
GuidedMatcher::get_keypoints_in_cell(const std::vector<cv::KeyPoint>& undist_keypts,
                                     const CellIndices& keypt_indices_in_cells,
                                     const float ref_x, const float ref_y, const float margin,
                                     const int min_level, const int max_level) const
{
    std::vector<size_t> indices;
    indices.reserve(undist_keypts.size());
    const int min_cell_idx_x = std::max(0, cvFloor((ref_x - grid_params_.img_min_width_ - margin) * grid_params_.inv_cell_width_));

    if (static_cast<int>(grid_params_.grid_cols_) <= min_cell_idx_x) {
        return indices;
    }

    const int max_cell_idx_x = std::min(static_cast<int>(grid_params_.grid_cols_ - 1), cvCeil((ref_x - grid_params_.img_min_width_ + margin) * grid_params_.inv_cell_width_));
    if (max_cell_idx_x < 0) {
        return indices;
    }

    const int min_cell_idx_y = std::max(0, cvFloor((ref_y - grid_params_.img_min_height_ - margin) * grid_params_.inv_cell_height_));
    if (static_cast<int>(grid_params_.grid_rows_) <= min_cell_idx_y) {
        return indices;
    }

    const int max_cell_idx_y = std::min(static_cast<int>(grid_params_.grid_rows_ - 1), cvCeil((ref_y - grid_params_.img_min_height_ + margin) * grid_params_.inv_cell_height_));
    if (max_cell_idx_y < 0) {
        return indices;
    }

    const bool check_level = (0 < min_level) || (0 <= max_level);
    for (int cell_idx_x = min_cell_idx_x; cell_idx_x <= max_cell_idx_x; ++cell_idx_x) {
        for (int cell_idx_y = min_cell_idx_y; cell_idx_y <= max_cell_idx_y; ++cell_idx_y) {
            const auto& keypt_indices_in_cell = keypt_indices_in_cells.at(cell_idx_x).at(cell_idx_y);

            if (keypt_indices_in_cell.empty()) {
                continue;
            }

            for (unsigned int idx : keypt_indices_in_cell) {
                const auto& keypt = undist_keypts[idx];
                if (check_level) {
                    if (keypt.octave < min_level || (0 <= max_level && max_level < keypt.octave)) {
                        continue;
                    }
                }
                const float dist_x = keypt.pt.x - ref_x;
                const float dist_y = keypt.pt.y - ref_y;
                if (std::abs(dist_x) < margin && std::abs(dist_y) < margin) {
                    indices.push_back(idx);
                    if (idx >= undist_keypts.size())
                    {
                        std::cout << "keypts idx error!" << idx << std::endl;
                        exit(0);
                    }
                }
            }
        }
    }
    return indices;
}

std::vector<cslam::Landmark*>
GuidedMatcher::update_local_landmarks(const std::vector<cslam::KeyFrame*>& local_keyframes, const size_t curr_frm_id)
{
    std::vector<cslam::Landmark*> local_landmarks;
    for (auto keyframe : local_keyframes)
    {
        for (auto lm : keyframe->landmarks_)
        {
            if (lm == nullptr) continue;
            // do not add twice
            if (lm->identifier_in_local_map_update_ == curr_frm_id) continue;
            lm->identifier_in_local_map_update_ = curr_frm_id;
            local_landmarks.push_back(lm);
        }
    }
    return local_landmarks;
}
bool 
GuidedMatcher::can_observe(Landmark* lm, const Frame& frame, const float ray_cos_thr,
                           Eigen::Vector2f& reproj, size_t& pred_scale_level) const 
{
    const Eigen::Vector3f pos_w = lm->get_pos_in_world();
    const Eigen::Matrix4f T_cw = frame.getTcw();
    const Eigen::Matrix3f rot_cw = T_cw.block<3,3>(0,0);
    const Eigen::Vector3f trans_cw = T_cw.block<3,1>(0,3);
    // camera_.reproject_to_image()
    const bool in_image = camera_.reproject_to_image(rot_cw, trans_cw, pos_w, grid_params_, reproj);
    if (!in_image) {
        return false;
    }
    // const Eigen::Vector3f cam_center = frame.get_cam_center();
    const Eigen::Vector3f cam_to_lm_vec = pos_w - frame.get_cam_center();
    const auto cam_to_lm_dist = cam_to_lm_vec.norm();
    if (!lm->is_inside_in_orb_scale(cam_to_lm_dist)) {
        return false;
    }

    const Eigen::Vector3f obs_mean_normal = lm->get_obs_mean_normal();
    const auto ray_cos = cam_to_lm_vec.dot(obs_mean_normal) / cam_to_lm_dist;
    if (ray_cos < ray_cos_thr) {
        return false;
    }

    pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, frame);
    return true;
}

size_t
GuidedMatcher::search_local_landmarks(std::vector<Landmark*>& local_landmarks, Frame& curr_frm)
{
    //go through the landmarks of the current frame
    for (auto lm : curr_frm.landmarks_)
    {
        //select the ones with a global landmark match
        if (lm != nullptr)
        {
            lm->is_observable_in_tracking_ = false;
            lm->identifier_in_local_lm_search_ = curr_frm.frame_id;
            lm->increase_num_observable();
        }
    }
    bool found_proj_candidate = false;
    // temporary variables
    Eigen::Vector2f reproj;
    size_t pred_scale_level;
    for (auto lm : local_landmarks) 
    {
        // avoid the landmarks which cannot be reprojected (== observed in the current frame)
        if (lm->identifier_in_local_lm_search_ == curr_frm.frame_id) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        // check the observability
        // if (curr_frm.can_observe(lm, 0.5, reproj, x_right, pred_scale_level)) {
        if (can_observe(lm, curr_frm, 0.5, reproj, pred_scale_level)) 
        {
            // pass the temporary variables
            lm->reproj_in_tracking_ = reproj;
            // lm->x_right_in_tracking_ = x_right;
            lm->scale_level_in_tracking_ = pred_scale_level;

            // this landmark can be reprojected
            lm->is_observable_in_tracking_ = true;

            // this landmark is observable from the current frame
            lm->increase_num_observable();

            found_proj_candidate = true;
        }
        else {
            // this landmark cannot be reprojected
            lm->is_observable_in_tracking_ = false;
        }
    }

    if (!found_proj_candidate) {
        return 0;
    }
    //TODO: margin depends on relocalisation
    constexpr float margin{5};
    return match_frame_and_landmarks(curr_frm, local_landmarks, margin);
}

size_t
// GuidedMatcher::match_frame_and_landmarks(const std::vector<float>& scale_factors, cslam::Frame& frm, std::vector<cslam::Landmark*>& local_landmarks, const float margin)
GuidedMatcher::match_frame_and_landmarks(cslam::Frame& frm, std::vector<cslam::Landmark*>& local_landmarks, const float margin)
{
    size_t num_matches{0};
    std::cout << "scale: " << frm.scale_factors_.size() << " local_lm: " << local_landmarks.size() << std::endl;

    for (auto local_lm : local_landmarks) {
        if (!local_lm->is_observable_in_tracking_) {
            continue;
        }
        if (local_lm->will_be_erased()) {
            continue;
        }
        // orb_params.scale_factors_.at 
        const auto pred_scale_level = local_lm->scale_level_in_tracking_;
        // std::cout << " local_lm->reproj_in_tracking_: " <<  local_lm->reproj_in_tracking_ << ", " << pred_scale_level << std::endl;
        // std::cout << frm.undist_keypts_.size() << "/" << frm.keypts_indices_in_cells_.size() << "/" << frm.scale_factors_.size() << std::endl;
        // Get the feature point of the area where the 3D point reprojects to
        const auto indices_in_cell = get_keypoints_in_cell(frm.undist_keypts_, frm.keypts_indices_in_cells_,
                                                           local_lm->reproj_in_tracking_(0), local_lm->reproj_in_tracking_(1),
                                                           margin * frm.scale_factors_.at(pred_scale_level),
                                                           pred_scale_level - 1, pred_scale_level);
        if (indices_in_cell.empty()) {
            continue;
        }

        const cv::Mat lm_desc = local_lm->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        int best_scale_level = -1;
        unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
        int second_best_scale_level = -1;
        int best_idx = -1;

        for (const auto idx : indices_in_cell) {
            //make sure that the first association of a landmark stays
            //continue if it already has a landmark
            if (frm.landmarks_.at(idx) && frm.landmarks_.at(idx)->has_observation()) {
                continue;
            }

            // if (0 < frm.stereo_x_right_.at(idx)) {
            //     const auto reproj_error = std::abs(local_lm->x_right_in_tracking_ - frm.stereo_x_right_.at(idx));
            //     if (margin * scale_factors.at(pred_scale_level) < reproj_error) {
            //         continue;
            //     }
            // }

            const cv::Mat& desc = frm.descriptors_.row(idx);

            const auto dist = compute_descriptor_distance_32(lm_desc, desc);

            if (dist < best_hamm_dist) {
                second_best_hamm_dist = best_hamm_dist;
                best_hamm_dist = dist;
                second_best_scale_level = best_scale_level;
                best_scale_level = frm.undist_keypts_.at(idx).octave;
                best_idx = idx;
            }
            else if (dist < second_best_hamm_dist) {
                second_best_scale_level = frm.undist_keypts_.at(idx).octave;
                second_best_hamm_dist = dist;
            }
        }
        constexpr float lowe_ratio{0.8};
        if (best_hamm_dist <= HAMMING_DIST_THR_HIGH) {
            // lowe's ratio test
            if (best_scale_level == second_best_scale_level && best_hamm_dist > lowe_ratio * second_best_hamm_dist) {
                continue;
            }

            // 対応情報を追加
            // add support information
            frm.landmarks_.at(best_idx) = local_lm;
            ++num_matches;
        }
    }

    return num_matches;
}


// size_t
std::vector<std::pair<size_t, size_t>>
GuidedMatcher::match_current_and_last_frame(cslam::Frame& curr_frm, const cslam::Frame& last_frm, const float margin) {
    size_t num_matches = 0;
    constexpr auto check_orientation_{true};
    constexpr float lowe_ratio_{0.9};
    openvslam::match::angle_checker<int> angle_checker;

    const Eigen::Matrix4f cam_pose_cw = curr_frm.getPose().inverse();
    const Eigen::Matrix3f rot_cw = cam_pose_cw.block<3, 3>(0, 0);
    const Eigen::Vector3f trans_cw = cam_pose_cw.block<3, 1>(0, 3);

    // const Eigen::Vector3f trans_wc = -rot_cw.transpose() * trans_cw;

    // const Eigen::Matrix3f rot_lw = last_frm.cam_pose_cw_.block<3, 3>(0, 0);
    // const Eigen::Vector3f trans_lw = last_frm.cam_pose_cw_.block<3, 1>(0, 3);

    // const Eigen::Vector3f trans_lc = rot_lw * trans_wc + trans_lw;

    // monocular以外の場合は，current->lastの並進ベクトルのz成分で前進しているか判断しているかを判定する
    // z成分が正に振れている -> 前進している

    // If it is not monocular, determine whether it is moving forward with the z component of the current-> last translation vector
    // z component is swinging positive-> moving forward
    // const bool assume_forward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
    //                             ? false : trans_lc(2) > curr_frm.camera_->true_baseline_;
    // // z成分が負に振れている -> 後退している
    // // z component swings negative-> retreats
    // const bool assume_backward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
    //                              ? false : -trans_lc(2) > curr_frm.camera_->true_baseline_;

    // For monocular forward/backward is always false
    // last frameの特徴点と対応が取れている3次元点を，current frameに再投影して対応を求める
    // Find the correspondence by reprojecting the 3D points that correspond to the feature points of the last frame to the current frame
    std::cout << "last_frm: " << last_frm.im_name << " nK: " << last_frm.num_keypts_<< "," << last_frm.landmarks_.size() << std::endl;
    std::vector<std::pair<size_t, size_t>> matches;
    for (unsigned int idx_last = 0; idx_last < last_frm.num_keypts_; ++idx_last) {
        auto lm = last_frm.landmarks_.at(idx_last);
        // 3次元点と対応が取れていない
        // std::cout << "lm: " << lm << "idx_last: " << idx_last << std::endl;
        // Not compatible with 3D points
        if (!lm) {
            continue;
        }
        // pose optimizationでoutlierになったものとは対応を取らない        
        // Does not correspond to the outlier in pose optimization
        if (last_frm.outlier_flags_.at(idx_last)) {
            continue;
        }
        

        // グローバル基準の3次元点座標
        // Global standard 3D point coordinates
        const Eigen::Vector3f pos_w = lm->get_pos_in_world();

        // 再投影して可視性を求める
        // Reproject to find visibility
        Eigen::Vector2f pt2D;


        // float x_right;
        // const bool in_image = camera_->reproject_to_image(rot_cw, trans_cw, pos_w, pt2D, x_right);
        // const bool in_image = camera_.reproject_to_image(rot_cw, trans_cw, pos_w, pt2D);

        // std::cout << "pos_w: " << pos_w << std::endl;
        // 画像外に再投影される場合はスルー
        // Thru if reprojected outside image
        if (!camera_.reproject_to_image(rot_cw, trans_cw, pos_w, grid_params_, pt2D))
        { 
            // std::cout << " out pt2D: " << pt2D.transpose() << std::endl;
            continue;
        }
        // std::cout << "in pt2D: " << pt2D.transpose() << std::endl;
        // 隣接フレーム間では対応する特徴点のスケールは一定であると仮定し，探索範囲を設定
        // Set search range assuming that the scale of corresponding feature points is constant between adjacent frames
        const auto last_scale_level = last_frm.keypts_.at(idx_last).octave;
        // std::cout << "in last_scale_level: " << last_scale_level  
                //   << "ud kpts: " << curr_frm.undist_keypts_.size() << std::endl;

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        // Get the feature point of the cell where the reprojected 3D point exists
        // std::vector<unsigned int> indices;
        // if (assume_forward) {
        //     indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
        //                                              margin * curr_frm.scale_factors_.at(last_scale_level),
        //                                              last_scale_level, last_frm.num_scale_levels_ - 1);
        // }
        // else if (assume_backward) {
        //     indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
        //                                              margin * curr_frm.scale_factors_.at(last_scale_level),
        //                                              0, last_scale_level);
        // }
        // else {
        const auto indices = get_keypoints_in_cell(curr_frm.undist_keypts_, curr_frm.keypts_indices_in_cells_, pt2D[0], pt2D[1],
                                                   margin * curr_frm.scale_factors_.at(last_scale_level),
                                                   last_scale_level -1, last_scale_level + 1);
        // const auto indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                            //  margin * curr_frm.scale_factors_.at(last_scale_level),
                                                            //  last_scale_level - 1, last_scale_level + 1);
        // }
        // std::cout << "indices: " << indices.size();
        if (indices.empty()) {
            continue;
        }

        const auto lm_desc = lm->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        int best_idx = -1;

        for (const auto curr_idx : indices) {
            // std::cout << "curr_idx: " << curr_idx << ", " << curr_frm.landmarks_.size() 
                    //   << "," << curr_frm.keypts_.size() << "," << curr_frm.undist_keypts_.size() << std::endl;
            //prevent adding new landmarks
            if (curr_frm.landmarks_.at(curr_idx) && curr_frm.landmarks_[curr_idx]->has_observation()) {
                continue;
            }
            // std::cout << "aft curr_idx: " << curr_idx << std::endl;
            //filter reprojection errors
            // if (curr_frm.stereo_x_right_.at(curr_idx) > 0) {
            //     const float reproj_error = std::fabs(x_right - curr_frm.stereo_x_right_.at(curr_idx));
            //     if (margin * curr_frm.scale_factors_.at(last_scale_level) < reproj_error) {
            //         continue;
            //     }
            // }

            const auto& desc = curr_frm.descriptors_.row(curr_idx);
            // std::cout << "desc: " << curr_idx << std::endl;
            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_hamm_dist) {
                best_hamm_dist = hamm_dist;
                best_idx = curr_idx;
            }
        }

        if (HAMMING_DIST_THR_HIGH < best_hamm_dist) {
            continue;
        }

        // 有効なmatchingとする
        // Valid matching
        curr_frm.landmarks_.at(best_idx) = lm;
        ++num_matches;

        if (check_orientation_) {
            const auto delta_angle
                    = last_frm.undist_keypts_.at(idx_last).angle - curr_frm.undist_keypts_.at(best_idx).angle;
            angle_checker.append_delta_angle(delta_angle, best_idx);
        }
        matches.emplace_back(idx_last,best_idx);
    }

    if (check_orientation_) {
        const auto invalid_matches = angle_checker.get_invalid_matches();
        for (const auto invalid_idx : invalid_matches) {
            curr_frm.landmarks_.at(invalid_idx) = nullptr;
            --num_matches;
        }
    }
    // return num_matches;
    return matches;
}
MatchIndices
GuidedMatcher::match_for_triangulation(const KeyFrame& kf1, const KeyFrame& kf2, const Eigen::Matrix3f& E_12) const
{
    // get the already matched landmarks
    const auto& lms1 = kf1.landmarks_;
    const auto& lms2 = kf2.landmarks_;
    const Eigen::Vector3f cam_center_1 = kf1.get_cam_center();
    const Eigen::Matrix4f T_cw = kf2.getTcw();
    const Eigen::Matrix3f rot_2w = T_cw.block<3,3>(0,0);
    const Eigen::Vector3f trans_2w = T_cw.block<3,1>(0,3);
    size_t num_matches{0};
    // matching情報を格納する
    // keyframe1の各特徴点に対して対応を求めるため，keyframe2で既にkeyframe1と対応が取れているものを除外するようにする
    std::vector<bool> is_already_matched_in_keyfrm_2(lms2.size(), false); // to keep track of potentially triangulated points

    // keyframe1のidxと対応しているkeyframe2のidxを格納する
    std::vector<int> matched_indices_2_in_keyfrm_1(lms1.size(), -1);
    MatchIndices matches;
    
    Eigen::Vector3f epiplane_in_keyfrm_2;
    camera_.reproject_to_bearing(rot_2w, trans_2w, cam_center_1, grid_params_, epiplane_in_keyfrm_2);
    // TODO: Maybe do some bag of words matching to reduce the number of potential keyframes!
    // For now, try exhaustive matching

    for (size_t idx_1 = 0; idx_1 < lms1.size(); ++idx_1)
    {
        const auto* lm_1 = lms1[idx_1];
        if (lm_1 != nullptr) continue; // already matched to a landmark

        auto best_hamm_dist = HAMMING_DIST_THR_LOW;
        int best_idx_2 = -1;
        const auto& keypt_1 = kf1.undist_keypts_.at(idx_1);
        const Eigen::Vector3f& bearing_1 = kf1.bearings_.at(idx_1);
        const auto& desc_1 = kf1.descriptors_.row(idx_1);

        //start exhaustive matching
        for (size_t idx_2 = 0; idx_2 < lms2.size(); ++idx_2)
        {
            const auto* lm_2 = lms2[idx_2];
            if (lm_2 != nullptr) continue; // already matched to a lm and not compatible
            if (is_already_matched_in_keyfrm_2.at(idx_2)) continue; //already matched to another feature


            // 特徴点・特徴量を取得
            // std::cout << "bearings: " << kf2.bearings_.size() << "/" << idx_2 << std::endl;
            const Eigen::Vector3f& bearing_2 = kf2.bearings_.at(idx_2);
            const auto& desc_2 = kf2.descriptors_.row(idx_2);
            const auto hamm_dist = compute_descriptor_distance_32(desc_1, desc_2);

            if (HAMMING_DIST_THR_LOW < hamm_dist || best_hamm_dist < hamm_dist) {
                continue;
            }

            // if (!is_stereo_keypt_1 && !is_stereo_keypt_2) {
                // If both are not stereo keypoints, don't use feature points near epipole
                const auto cos_dist = epiplane_in_keyfrm_2.dot(bearing_2);
                // Threshold angle between epipole and bearing (= 3.0deg)
                constexpr double cos_dist_thr = 0.99862953475;
                // do not match if the included angle is smaller than the threshold
                if (cos_dist_thr < cos_dist) { 
                    continue; 
                    }
            // }
            

            // E行列による整合性チェック
            const bool is_inlier = check_epipolar_constraint(bearing_1, bearing_2, E_12,
                                                             kf1.scale_factors_.at(keypt_1.octave));
            if (is_inlier) {
                best_idx_2 = idx_2;
                best_hamm_dist = hamm_dist;
            }
        }
        if (best_idx_2 < 0) {
            continue;
        }

        is_already_matched_in_keyfrm_2.at(best_idx_2) = true;
        matched_indices_2_in_keyfrm_1.at(idx_1) = best_idx_2;
        ++num_matches;
        // no angle checker for now
        // if (check_orientation_) {
        //             const auto delta_angle
        //                 = keypt_1.angle - keyfrm_2->undist_keypts_.at(best_idx_2).angle;
        //             angle_checker.append_delta_angle(delta_angle, idx_1);
        //         }
    }

    matches.reserve(num_matches);
    //We do not check the orientation
    for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_keyfrm_1.size(); ++idx_1) {
        if (matched_indices_2_in_keyfrm_1.at(idx_1) < 0) {
            continue;
        }
        matches.emplace_back(std::make_pair(idx_1, matched_indices_2_in_keyfrm_1.at(idx_1)));
    }
    return matches;
}

bool 
GuidedMatcher::check_epipolar_constraint(const Eigen::Vector3f& bearing_1, const Eigen::Vector3f& bearing_2,
                                       const Eigen::Matrix3f& E_12, const float bearing_1_scale_factor) 
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

template<typename T> size_t 
GuidedMatcher::replace_duplication(KeyFrame* keyfrm, const T& landmarks_to_check, const float margin)  const
{
    unsigned int num_fused = 0;

    const Eigen::Matrix3f rot_cw = keyfrm->get_rotation();
    const Eigen::Vector3f trans_cw = keyfrm->get_translation();
    const Eigen::Vector3f cam_center = keyfrm->get_cam_center();

    for (const auto lm : landmarks_to_check) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        std::cout << "lm is_observed_in_keyframe: " << lm->lm_id_ << std::endl;
        if (lm->is_observed_in_keyframe(keyfrm)) {
            continue;
        }
        std::cout << "lm: " << lm->lm_id_ << std::endl;
        // グローバル基準の3次元点座標
        const Eigen::Vector3f pos_w = lm->get_pos_in_world();
        // 再投影して可視性を求める
        Eigen::Vector2f reproj;
        float x_right;
        // const bool in_image = camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);
        const bool in_image = camera_.reproject_to_image(rot_cw, trans_cw, pos_w, grid_params_, reproj);
        std::cout << "rot_cw: " << rot_cw << "/" << trans_cw << "pos: " << pos_w << " reproj: " << reproj << std::endl;
        std::cout << "K_eig: " << camera_.K_pixel_eig << " grid: "
                  << grid_params_.img_min_height_ << "/" << grid_params_.img_max_height_ << "/" 
                  << grid_params_.img_min_width_ << "/" << grid_params_.img_max_width_ 
                  << std::endl;
        // 画像外に再投影される場合はスルー
        if (!in_image) {
            continue;
        }

        // ORBスケールの範囲内であることを確認
        const Eigen::Vector3f cam_to_lm_vec = pos_w - cam_center;
        const auto cam_to_lm_dist = cam_to_lm_vec.norm();
        const auto max_cam_to_lm_dist = lm->get_max_valid_distance();
        const auto min_cam_to_lm_dist = lm->get_min_valid_distance();

        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        std::cout << "lm get_obs_mean_normal: " << lm->lm_id_ << std::endl;
        // 3次元点の平均観測ベクトルとの角度を計算し，閾値(60deg)より大きければ破棄
        const Eigen::Vector3f obs_mean_normal = lm->get_obs_mean_normal();

        if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist) {
            continue;
        }

        // 3次元点を再投影した点が存在するcellの特徴点を取得
        const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, *keyfrm);
        std::cout << "lm get_keypoints_in_cell: " << lm->lm_id_  
                  << " pred: " << pred_scale_level << "/" << reproj.transpose() << ", size: " << keyfrm->keypts_indices_in_cells_.size() 
                  << ", " << margin*keyfrm->scale_factors_.at(pred_scale_level)<< std::endl;

        // const auto indices = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->scale_factors_.at(pred_scale_level));
        const auto indices = get_keypoints_in_cell(keyfrm->undist_keypts_, keyfrm->keypts_indices_in_cells_, 
                                                   reproj(0), reproj(1), margin*keyfrm->scale_factors_.at(pred_scale_level));//, scale_1, scale_1);

        if (indices.empty()) {
            continue;
        }
        std::cout << "lm get_descriptor: " << lm->lm_id_ << std::endl;

        // descriptorが最も近い特徴点を探す
        const auto lm_desc = lm->get_descriptor();

        unsigned int best_dist = MAX_HAMMING_DIST;
        int best_idx = -1;

        for (const auto idx : indices) {
            const auto& keypt = keyfrm->undist_keypts_.at(idx);

            const auto scale_level = static_cast<unsigned int>(keypt.octave);

            // TODO: keyfrm->get_keypts_in_cell()でスケールの判断をする
            if (scale_level < pred_scale_level - 1 || pred_scale_level < scale_level) {
                continue;
            }

            // if (keyfrm->stereo_x_right_.at(idx) >= 0) {
            //     // stereo matchが存在する場合は自由度3の再投影誤差を計算する
            //     // If there is a stereo match, calculate the reprojection error with 3 degrees of freedom
            //     const auto e_x = reproj(0) - keypt.pt.x;
            //     const auto e_y = reproj(1) - keypt.pt.y;
            //     const auto e_x_right = x_right - keyfrm->stereo_x_right_.at(idx);
            //     const auto reproj_error_sq = e_x * e_x + e_y * e_y + e_x_right * e_x_right;

            //     // 自由度n=3
            //     constexpr float chi_sq_3D = 7.81473;
            //     if (chi_sq_3D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level)) {
            //         continue;
            //     }
            // }
            // else {
                // stereo matchが存在しない場合は自由度2の再投影誤差を計算する
                // If there is no stereo match, calculate the reprojection error with 2 degrees of freedom
                const auto e_x = reproj(0) - keypt.pt.x;
                const auto e_y = reproj(1) - keypt.pt.y;
                const auto reproj_error_sq = e_x * e_x + e_y * e_y;

                // 自由度n=2
                constexpr float chi_sq_2D = 5.99146;
                if (chi_sq_2D < reproj_error_sq * keyfrm->inv_level_sigma_sq_.at(scale_level)) {
                    continue;
                }
            // }

            const auto& desc = keyfrm->descriptors_.row(idx);

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_dist) {
                best_dist = hamm_dist;
                best_idx = idx;
            }
        }

        if (HAMMING_DIST_THR_LOW < best_dist) {
            continue;
        }

        auto* lm_in_keyfrm = keyfrm->landmarks_.at(best_idx);//keyfrm->get_landmark(best_idx);
        if (lm_in_keyfrm) {
            // keyframeのbest_idxに対応する3次元点が存在する -> 重複している場合
            if (!lm_in_keyfrm->will_be_erased()) {
                // より信頼できる(=観測数が多い)3次元点で置き換える
                if (lm->num_observations() < lm_in_keyfrm->num_observations()) {
                    // lm_in_keyfrmで置き換える
                    lm->replace(lm_in_keyfrm);
                }
                else {
                    // lmで置き換える
                    lm_in_keyfrm->replace(lm);
                }
            }
        }
        else {
            // keyframeのbest_idxに対応する3次元点が存在しない
            // 観測情報を追加
            lm->add_observation(keyfrm, best_idx);
            keyfrm->add_landmark(lm, best_idx);
        }

        ++num_fused;
    }

    return num_fused;
}

template size_t GuidedMatcher::replace_duplication(KeyFrame*, const std::vector<Landmark*>&, const float) const;
template size_t GuidedMatcher::replace_duplication(KeyFrame*, const std::unordered_set<Landmark*>&, const float) const;

};