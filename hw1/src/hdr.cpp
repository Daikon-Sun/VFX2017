#include "hdr.hpp"

void DEBEVEC::process(Mat& result, float lambda) {
  int pic_num = (int)_pics.size();
  int sam_num = (int)_points.size();
  int num_channels = _pics[0].channels();
  vector<Mat> X(num_channels);
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
    solve(A, B, X[i], DECOMP_SVD);
  }

  Mat res[3];
  for(int i = 0; i<3; ++i) res[i] = Mat::zeros(_pics[0].size(), CV_32F);
  for(int c = 0; c<num_channels; ++c) {
    for(int i = 0; i<res[c].cols; ++i) for(int j = 0; j<res[c].rows; ++j) {
      float& sum = res[c].at<float>(j, i);
      float sum_w = 0;
      for(int k = 0; k<pic_num; ++k) {
        int val = _pics[k].at<Vec3b>(j, i)[c];
        int w = W(val); 
        sum += w * (X[c].at<float>(0, val) - log(_etimes[k]));
        sum_w += w;
      }
      sum = exp(sum/sum_w);
    }
  }
  merge(res, 3, result);
}
