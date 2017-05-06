#include "util.hpp"

#ifndef _DETECTION_HPP_
#define _DETECTION_HPP_

const Mat Kernel_x = (Mat_<float>(1, 3) << -0.5, 0, 0.5); 
const Mat Kernel_y = (Mat_<float>(3, 1) << -0.5, 0, 0.5); 

class DETECTION {
public:
  DETECTION(const vector<Mat>& i, vector< vector<Keypoint> >& k) 
    : imgs(i), keypoints(k) {};
  void MSOP();

protected:
  const vector<Mat>& imgs;
  vector< vector<Keypoint> >& keypoints;
};

#endif
