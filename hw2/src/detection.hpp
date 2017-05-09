#include "util.hpp"

#ifndef _DETECTION_HPP_
#define _DETECTION_HPP_

const Mat Kernel_x = (Mat_<double>(1, 3) << -0.5, 0, 0.5); 
const Mat Kernel_y = (Mat_<double>(3, 1) << -0.5, 0, 0.5); 

constexpr double rt8 = 1/sqrt(8);
constexpr double rt2 = 1/sqrt(2);
const Mat haar = 
  (Mat_<float>(8, 8) << rt8, rt8, rt8, rt8, rt8, rt8, rt8, rt8,
                        rt8, rt8, rt8, rt8,-rt8,-rt8,-rt8,-rt8,
                        0.5, 0.5,-0.5,-0.5,   0,   0,   0,   0,
                          0,   0,   0,   0, 0.5, 0.5,-0.5,-0.5,
                        rt2,-rt2,   0,   0,   0,   0,   0,   0,
                          0,   0, rt2,-rt2,   0,   0,   0,   0,
                          0,   0,   0,   0, rt2,-rt2,   0,   0,
                          0,   0,   0,   0,   0,   0, rt2,-rt2);
const Mat haar_T = [&]() -> const Mat { Mat res; transpose(haar, res);
                                        return res.clone(); }();

class DETECTION {
public:
  DETECTION(const vector<Mat>& i, vector< vector<Keypoint> >& k) 
    : imgs(i), keypoints(k) {};
  void MSOP();
  void SIFT();
private:
  bool is_extrema(const vector< vector<Mat> >&, int, int, int, int);
  const vector<Mat>& imgs;
  vector< vector<Keypoint> >& keypoints;

};

#endif
