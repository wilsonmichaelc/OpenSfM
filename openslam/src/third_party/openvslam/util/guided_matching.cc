#include "guided_matching.h"
#include "angle_checker.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
namespace guided_matching
{

GridParameters::GridParameters(unsigned int grid_cols_, unsigned int grid_rows_,
                float img_min_width_, float img_min_height_,
                float inv_cell_width_, float inv_cell_height_):
                grid_cols(grid_cols_), grid_rows(grid_rows_),
                img_min_width(img_min_width_), img_min_height(img_min_height_),
                inv_cell_width(inv_cell_width_), inv_cell_height(inv_cell_height_)
                {}


// void
// get_num_scale_levels

void
assign_points_to_grid(const GridParameters& params, const Eigen::MatrixXf& undist_keypts, CellIndices& keypt_indices_in_cells)
{
    const size_t num_pts = undist_keypts.rows();
    const size_t num_to_reserve = 0.5 * num_pts / (params.grid_cols*params.grid_rows);
    keypt_indices_in_cells.resize(params.grid_cols);
    for (auto& keypt_indices_in_row : keypt_indices_in_cells) {
        keypt_indices_in_row.resize(params.grid_rows);
        for (auto& keypt_indices_in_cell : keypt_indices_in_row) {
            keypt_indices_in_cell.reserve(num_to_reserve);
        }
    }
    for (size_t idx = 0; idx < num_pts; ++idx) {
        // const auto& keypt = undist_keypts.at(idx);
        const Eigen::Vector2f pt = undist_keypts.block<1,2>(idx,0);
        // std::cout << "pt: " << pt.transpose() << std::endl;
        const int cell_idx_x = std::round((pt[0] - params.img_min_width) * params.inv_cell_width);
        const int cell_idx_y = std::round((pt[1] - params.img_min_height) * params.inv_cell_height);
        if ((0 <= cell_idx_x && cell_idx_x < static_cast<int>(params.grid_cols)
            && 0 <= cell_idx_y && cell_idx_y < static_cast<int>(params.grid_rows)))
        {
            keypt_indices_in_cells.at(cell_idx_x).at(cell_idx_y).push_back(idx);
        }
    }
}
CellIndices 
assign_keypoints_to_grid(const GridParameters& params, const Eigen::MatrixXf& undist_keypts) {
    CellIndices keypt_indices_in_cells;
    assign_points_to_grid(params, undist_keypts, keypt_indices_in_cells);
    return keypt_indices_in_cells;
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

static constexpr unsigned int HAMMING_DIST_THR_LOW = 50;
static constexpr unsigned int HAMMING_DIST_THR_HIGH = 100;
// static constexpr unsigned int HAMMING_DIST_THR_LOW = 80;
// static constexpr unsigned int HAMMING_DIST_THR_HIGH = 120;
static constexpr unsigned int MAX_HAMMING_DIST = 256;

//! ORB特徴量間のハミング距離を計算する
inline unsigned int compute_descriptor_distance_32(const cv::Mat& desc_1, const cv::Mat& desc_2) {
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    constexpr uint32_t mask_1 = 0x55555555U;
    constexpr uint32_t mask_2 = 0x33333333U;
    constexpr uint32_t mask_3 = 0x0F0F0F0FU;
    constexpr uint32_t mask_4 = 0x01010101U;

    const auto* pa = desc_1.ptr<uint32_t>();
    const auto* pb = desc_2.ptr<uint32_t>();

    unsigned int dist = 0;

    for (unsigned int i = 0; i < 8; ++i, ++pa, ++pb) {
        auto v = *pa ^*pb;
        v -= ((v >> 1) & mask_1);
        v = (v & mask_2) + ((v >> 2) & mask_2);
        dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 24;
    }

    return dist;
}


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
MatchIndices
match_frame_to_frame(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
                     const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
                     const GridParameters& grid_params, const size_t margin)
{
    constexpr auto check_orientation_{true};
    constexpr float lowe_ratio_{0.9};
    // std::cout << "grid: " << grid_params.img_min_width << "/" << grid_params.img_min_height << " inv: " << grid_params.inv_cell_height << "," << grid_params.inv_cell_width
    //                       << grid_params.grid_cols << "/" << grid_params.grid_rows << std::endl; 
    // std::cout << "match_frame_to_frame" << std::endl;
    const size_t num_pts_1 = undist_keypts_1.rows();
    const size_t num_pts_2 = undist_keypts_2.rows();
    // std::cout << "match_frame_to_frame" << num_pts_1 << "/" << num_pts_2  
    //           << " desc: " << desc_1.rows() << ", " << desc_1.cols()  
    //           << " desc2: " << desc_2.rows() << ", " << desc_2.cols() << std::endl;
    MatchIndices matches; // Index in 1, Index in 2
    matches.reserve(num_pts_1);
    std::vector<unsigned int> matched_dists_in_frm_2(num_pts_2, MAX_HAMMING_DIST);
    std::vector<int> matched_indices_1_in_frm_2(num_pts_2, -1);
    std::vector<int> matched_indices_2_in_frm_1 = std::vector<int>(num_pts_1, -1);
    size_t num_matches = 0; // Todo: should be the same as matches.size()
    openvslam::match::angle_checker<int> angle_checker;
    // Wrap the descriptors in a CV Mat to make handling easier
    cv::Mat desc1_cv;
    cv::eigen2cv(desc_1, desc1_cv);
    cv::Mat desc2_cv;
    cv::eigen2cv(desc_2, desc2_cv);
    for (size_t idx_1 = 0; idx_1 < num_pts_1; ++idx_1)
    {
        // f1 = x, y, size, angle, octave
        const OrbFeature f1 = undist_keypts_1.block<1,5>(idx_1,0);
        const float scale_1 = f1[4];
        // std::cout << "f1: " << f1.transpose() << std::endl;
        if (scale_1 < 0) continue;
        // Now, 
        const auto indices = get_keypoints_in_cell(grid_params, undist_keypts_2, cell_indices_2, f1[0], f1[1], margin, scale_1, scale_1);
        // std::cout << "indices: " << indices.size() << std::endl;
        if (indices.empty()) continue; // No valid match
        // std::cout << "indices: " << indices.size() << std::endl;

        // Read the descriptor
        const auto& d1 = desc1_cv.row(idx_1);
        auto best_hamm_dist = MAX_HAMMING_DIST;
        auto second_best_hamm_dist = MAX_HAMMING_DIST;
        int best_idx_2 = -1;
        for (const auto idx_2 : indices) 
        {
            const auto& d2 = desc2_cv.row(idx_2);
            const auto hamm_dist = compute_descriptor_distance_32(d1, d2);
            // std::cout << "d1: " << d1 << "\n d2: " << d2 << "=" << hamm_dist << std::endl;
            // through if the point already matched is closer
            if (matched_dists_in_frm_2.at(idx_2) <= hamm_dist) {
                // std::cout << "cont here1" << std::endl;
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
            // std::cout << "cont HAMMING_DIST_THR_LOW" << std::endl;
            continue;
        }

        // ratio test
        if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist)) {
            // std::cout << "cont lowe_ratio_" << std::endl;

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
        // std::cout << "num_matches: " << num_matches << std::endl;



        if (check_orientation_) {
            // const auto delta_angle
                    // = undist_keypts_1.at(idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
            const auto delta_angle
                    = undist_keypts_1(idx_1, 3) -  undist_keypts_2(best_idx_2, 3); // (idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
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
            std::cout << "Found match at: " << idx_1 << "/" << idx_2 << std::endl;
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

std::vector<size_t>
match_frame_to_frame_dbg(const Eigen::MatrixXf& undist_keypts_1, const Eigen::MatrixXf& undist_keypts_2,
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_1,
                     Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& desc_2,
                     const CellIndices& cell_indices_2, const Eigen::MatrixX2f& prevMatched,
                     const GridParameters& grid_params, const size_t margin)
{
    constexpr auto check_orientation_{true};
    constexpr float lowe_ratio_{0.9};
    std::cout << "grid: " << grid_params.img_min_width << "/" << grid_params.img_min_height << " inv: " << grid_params.inv_cell_height << "," << grid_params.inv_cell_width
                          << grid_params.grid_cols << "/" << grid_params.grid_rows << std::endl; 
    std::cout << "match_frame_to_frame" << std::endl;
    const size_t num_pts_1 = undist_keypts_1.rows();
    const size_t num_pts_2 = undist_keypts_2.rows();
    std::cout << "match_frame_to_frame" << num_pts_1 << "/" << num_pts_2  
              << " desc: " << desc_1.rows() << ", " << desc_1.cols()  
              << " desc2: " << desc_2.rows() << ", " << desc_2.cols() << std::endl;
    MatchIndices matches; // Index in 1, Index in 2
    matches.reserve(num_pts_1);
    std::vector<unsigned int> matched_dists_in_frm_2(num_pts_2, MAX_HAMMING_DIST);
    std::vector<int> matched_indices_1_in_frm_2(num_pts_2, -1);
    std::vector<int> matched_indices_2_in_frm_1 = std::vector<int>(num_pts_1, -1);
    size_t num_matches = 0; // Todo: should be the same as matches.size()
    openvslam::match::angle_checker<int> angle_checker;
    // Wrap the descriptors in a CV Mat to make handling easier
    cv::Mat desc1_cv(desc_1.rows(), desc_1.cols(), CV_8UC1, desc_1.data());
    cv::Mat desc2_cv(desc_2.rows(), desc_2.cols(), CV_8UC1, desc_2.data());
    
    for (size_t idx_1 = 0; idx_1 < num_pts_1; ++idx_1)
    {
        // f1 = x, y, size, angle, octave
        const OrbFeature f1 = undist_keypts_1.block<1,5>(idx_1,0);
        // std::cout << "idx_1: " << idx_1 << " f1: " << f1 << std::endl;
        const float scale_1 = f1[4];
        // std::cout << "f1: " << f1.transpose() << std::endl;
        if (scale_1 < 0) continue;
        // Now, 
        const auto indices = get_keypoints_in_cell(grid_params, undist_keypts_2, cell_indices_2, f1[0], f1[1], margin, scale_1, scale_1);
        std::cout << "indices: " << indices.size() << ", " << f1[0] << ", " << f1[1] << std::endl;
        if (indices.empty()) continue; // No valid match
        std::cout << "indices: " << indices.size() << "idx: " << idx_1 
                  << " f1: " << f1 << std::endl;
        for (const auto idx_2 : indices) 
        {
            std::cout << "idx_2: " << idx_2 <<": "
                      << undist_keypts_2.block<1,5>(idx_2,0) << std::endl;
        }
        // return indices;
        // Read the descriptor
        const auto& d1 = desc1_cv.row(idx_1);
        auto best_hamm_dist = MAX_HAMMING_DIST;
        auto second_best_hamm_dist = MAX_HAMMING_DIST;
        int best_idx_2 = -1;
        for (const auto idx_2 : indices) 
        {
            const auto& d2 = desc2_cv.row(idx_2);
            const auto hamm_dist = compute_descriptor_distance_32(d1, d2);
            std::cout << "d1: " << d1 << "\n d2: " << d2 << "=" << hamm_dist << std::endl;
            // through if the point already matched is closer
            if (matched_dists_in_frm_2.at(idx_2) <= hamm_dist) {
                std::cout << "cont here1" << std::endl;
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
            std::cout << "cont HAMMING_DIST_THR_LOW" << std::endl;
            continue;
        }

        // ratio test
        if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist)) {
            std::cout << "cont lowe_ratio_" << std::endl;

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
        std::cout << "num_matches: " << num_matches << std::endl;



        if (check_orientation_) {
            // const auto delta_angle
                    // = undist_keypts_1.at(idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
            const auto delta_angle
                    = undist_keypts_1(idx_1, 3) -  undist_keypts_2(best_idx_2, 3); // (idx_1).angle - undist_keypts_2.at(best_idx_2).angle;
            angle_checker.append_delta_angle(delta_angle, idx_1);
        }
        return indices;
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
            std::cout << "Found match at: " << idx_1 << "/" << idx_2 << std::endl;
        }
    } 

    // TODO: update this out of the loop!
    // previous matchesを更新する
    // for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
    //     if (0 <= matched_indices_2_in_frm_1.at(idx_1)) {
    //         prev_matched_pts.at(idx_1) = undist_keypts_2.at(matched_indices_2_in_frm_1.at(idx_1)).pt;
    //     }
    // }


    std::vector<size_t> dummy;
    return dummy;
};






void match_points_to_frame(){std::cout << "match_points_to_frame" << std::endl;};



std::vector<size_t> 
get_keypoints_in_cell(const GridParameters& grid_params, const Eigen::MatrixXf& undist_keypts,
                      const CellIndices& keypt_indices_in_cells,
                      const float ref_x, const float ref_y, const float margin,
                      const int min_level, const int max_level)
{
    std::vector<size_t> indices;
    indices.reserve(undist_keypts.size());
    const int min_cell_idx_x = std::max(0, cvFloor((ref_x - grid_params.img_min_width - margin) * grid_params.inv_cell_width));

    if (static_cast<int>(grid_params.grid_cols) <= min_cell_idx_x) {
        return indices;
    }

    const int max_cell_idx_x = std::min(static_cast<int>(grid_params.grid_cols - 1), cvCeil((ref_x - grid_params.img_min_width + margin) * grid_params.inv_cell_width));
    // std::cout << "max_cell_idx_x: " << max_cell_idx_x << std::endl;
    if (max_cell_idx_x < 0) {
        return indices;
    }

    const int min_cell_idx_y = std::max(0, cvFloor((ref_y - grid_params.img_min_height - margin) * grid_params.inv_cell_height));
    // std::cout << "min_cell_idx_y: " << min_cell_idx_y << std::endl;
    if (static_cast<int>(grid_params.grid_rows) <= min_cell_idx_y) {
        return indices;
    }

    const int max_cell_idx_y = std::min(static_cast<int>(grid_params.grid_rows- 1), cvCeil((ref_y - grid_params.img_min_height + margin) * grid_params.inv_cell_height));
    // std::cout << "max_cell_idx_y: " << max_cell_idx_y << std::endl;
    if (max_cell_idx_y < 0) {
        return indices;
    }

    const bool check_level = (0 < min_level) || (0 <= max_level);
    std::cout << "check_level: " << check_level << std::endl;
    for (int cell_idx_x = min_cell_idx_x; cell_idx_x <= max_cell_idx_x; ++cell_idx_x) {
        for (int cell_idx_y = min_cell_idx_y; cell_idx_y <= max_cell_idx_y; ++cell_idx_y) {
            const auto& keypt_indices_in_cell = keypt_indices_in_cells.at(cell_idx_x).at(cell_idx_y);
            // std::cout << "keypt_indices_in_cell: " << keypt_indices_in_cell.size() << std::endl;

            if (keypt_indices_in_cell.empty()) {
                continue;
            }

            for (unsigned int idx : keypt_indices_in_cell) {
                const OrbFeature feature = undist_keypts.block<1,5>(idx,0);
                // std::cout << " feature: " << feature << " ref: " << ref_x << "/" << ref_y << std::endl;
                const float octave = feature[4];
                if (check_level) {
                    if (octave < min_level || (0 <= max_level && max_level < octave)) {
                        // std::cout << "cont lvl not matching!" << "min_level: " << min_level << " max_level: " 
                        //   << max_level << " octave: " << octave << std::endl;
                        continue;
                    }
                }

                const float dist_x = feature[0] - ref_x;
                const float dist_y = feature[1] - ref_y;
                // std::cout << "dist: " << dist_x << "/" << dist_y << " ref: " << ref_x << "/" << ref_y << std::endl;
                if (std::abs(dist_x) < margin && std::abs(dist_y) < margin) {
                    indices.push_back(idx);
                    // std::cout << "idx: " << idx << std::endl;
                    // exit(0);
                }
            }
        }
    }

    return indices;
}
};