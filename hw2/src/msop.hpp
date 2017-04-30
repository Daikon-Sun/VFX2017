#ifndef _MSOP_HPP_
#define _MSOP_HPP_

#include "type.hpp"

const Mat Kernel_x = (Mat_<float>(1, 3) << -0.5, 0, 0.5); 
const Mat Kernel_y = (Mat_<float>(3, 1) << -0.5, 0, 0.5); 

constexpr float rt8 = 1/sqrt(8);
constexpr float rt2 = 1/sqrt(2);
const Mat HAAR = 
  (Mat_<float>(8, 8) << rt8, rt8, rt8, rt8, rt8, rt8, rt8, rt8,
                        rt8, rt8, rt8, rt8,-rt8,-rt8,-rt8,-rt8,
                        0.5, 0.5,-0.5,-0.5,   0,   0,   0,   0,
                          0,   0,   0,   0, 0.5, 0.5,-0.5,-0.5,
                        rt2,-rt2,   0,   0,   0,   0,   0,   0,
                          0,   0, rt2,-rt2,   0,   0,   0,   0,
                          0,   0,   0,   0, rt2,-rt2,   0,   0,
                          0,   0,   0,   0,   0,   0, rt2,-rt2);
const Mat HAAR_T = [&]() -> const Mat { Mat res; transpose(HAAR, res);
                                        return res.clone(); }();
class MSOP {
public:
  MSOP() {}
  void process(const vector<Mat>&);

private:
  void matching();
  vector< vector< vector<Keypoint> > > keypoints;
};

#endif
