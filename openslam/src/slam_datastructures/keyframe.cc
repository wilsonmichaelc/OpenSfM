#include "keyframe.h"
#include "frame.h"
namespace cslam
{
// // KeyFrame::KeyFrame(const size_t kf_id, const size_t src_frm_id):
//     kf_id_(kf_id), src_frm_id_(src_frm_id)
// {

// }

KeyFrame::KeyFrame(const size_t kf_id, const Frame& frame):
    kf_id_(kf_id), src_frm_id_(frame.mFrameId),
    keypts_(frame.mKeyPts), undist_keypts_(frame.undist_keypts_),
    bearings_(frame.bearings_), descriptors_(frame.descriptors_)
{

}

}