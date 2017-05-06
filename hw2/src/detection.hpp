#include "util.hpp"

#ifndef _DETECTION_HPP_
#define _DETECTION_HPP_

//extern float F;
//constexpr int BIN_NUM = 10;
//constexpr float BOUND = BIN_NUM/2.0-0.5;
//constexpr float Y_MAX_DIFF = 50;
//constexpr float THRESHOLD = 0.65;
//constexpr float F = 1500;
//constexpr int RANSAC_K = 4000;
//constexpr int RANSAC_THRESHOLD = 100;

const Mat Kernel_x = (Mat_<float>(1, 3) << -0.5, 0, 0.5); 
const Mat Kernel_y = (Mat_<float>(3, 1) << -0.5, 0, 0.5); 

class DETECTION {
public:
  DETECTION(const vector<Mat>& i, vector< vector<Keypoint> >& k) 
    : imgs(i), keypoints(k) {};
  void MSOP();
  //void SIFT(const vector<Mat>&, vector< vector<Keypoint> >&);

private:
  //void warping();
  //void RANSAC();
  //void blending();

  //pair<float, float> projected_xy(float, float, float, float);
  //bool in_mid(const int& x) { return x >= 0 && x < BIN_NUM-1; };
  //bool is_align(const Keypoint&, const Keypoint&);
  //bool check_match(const tuple<int, int, float>&, size_t, float) const;
  //bool is_inliner(size_t, float, float, pair<int, int>&);

  //vector< Mat > imgs;
protected:
  const vector<Mat>& imgs;
  vector< vector<Keypoint> >& keypoints;
  //vector< vector< pair<int, int> > > match_pairs;
  //vector< pair<float, float> > shift;
};

#endif
