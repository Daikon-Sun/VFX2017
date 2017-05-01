#ifndef _MSOP_HPP_
#define _MSOP_HPP_

#include "type.hpp"

constexpr int BIN_NUM = 10;
constexpr float BOUND = BIN_NUM/2.0-0.5;
constexpr float Y_MAX_DIFF = 50;
constexpr float THRESHOLD = 0.65;
constexpr float F = 4500;

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
  MSOP() : tot_kpts(0) {}
  void process(const vector<Mat>&);

private:
  void matching(const vector<Mat>&);
  void warp_image(const vector<Mat>&);
  bool in_mid(const int& x) { return x >= 0 && x < BIN_NUM-1; };
  bool is_align(const Keypoint&, const Keypoint&);
  bool check_match(const tuple<int, int, float>&, size_t, float) const;
  vector< vector<Keypoint> > keypoints;
  size_t tot_kpts;
};

#endif
