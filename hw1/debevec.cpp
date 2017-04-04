#include "debevec.hpp"

void DEBEVEC::process(vector<Mat>& results, float lambda) {
  int pic_num = (int)_pics.size();
  int sam_num = (int)_points.size();
  int num_channels = _pics[0].channels();
  results.resize(num_channels);
  for(int i = 0; i<num_channels; ++i) {
    Mat A = Mat::zeros(sam_num*pic_num + 257, 256 + sam_num, CV_64F);
    Mat B = Mat::zeros(A.rows, 1, CV_64F);

    int eq = 0;
    for(int j = 0; j<sam_num; ++j) for(int k = 0; k<pic_num; ++k) {
      int val = _pics[k].at<Vec3b>(_points[j])[i];
      int w = W(val);
      A.at<float>(eq, val) = w;
      A.at<float>(eq, 256+j) = -w;
      B.at<float>(eq, 0) = w * log(_etimes[k]);
      ++eq;
    }
    A.at<float>(eq, 128) = 1;
    ++eq;

    for(int j = 1; j<255; ++j) {
      int w = W(j);
      A.at<float>(eq, j-1) = lambda * w;
      A.at<float>(eq, j) = -2 * lambda * w;
      A.at<float>(eq, j+1) = lambda * w;
      ++eq;
    }
    solve(A, B, results[i], DECOMP_SVD);
  }
}
