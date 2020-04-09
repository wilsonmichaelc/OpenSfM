#include <slam/slam_utilities.h>
#include <Eigen/Core>
namespace slam
{
class PySlamUtilities
{
public:
static Eigen::MatrixXf GetUndistortedKeyptsFromShot(const map::Shot& shot)
{
    return SlamUtilities::convertOpenCVKptsToEigen(shot.slam_data_.undist_keypts_);
}

static Eigen::MatrixXf GetKeyptsFromShot(const map::Shot& shot)
{
    return SlamUtilities::convertOpenCVKptsToEigen(shot.GetKeyPoints());
}
};
}