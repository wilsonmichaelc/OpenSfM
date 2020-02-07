#include "keyframe.h"
#include "frame.h"
#include "landmark.h"
namespace cslam
{

KeyFrame::KeyFrame(const size_t kf_id, const Frame& frame):
    kf_id_(kf_id), src_frm_id_(frame.mFrameId),
    keypts_(frame.keypts_), undist_keypts_(frame.undist_keypts_),
    bearings_(frame.bearings_), descriptors_(frame.descriptors_),
    scale_factors_(frame.scale_factors_), num_scale_levels_(scale_factors_.size()),
    landmarks_(frame.landmarks_)
{

}

void
KeyFrame::add_landmark(Landmark* lm, const size_t idx)
{
    landmarks_[idx] = lm;
}
}