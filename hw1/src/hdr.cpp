#include "hdr.hpp"

void DEBEVEC::process(Mat& result, double lambda) {

  Mat W(1, 256, CV_64FC1);
  for(int i = 0; i<256; ++i) W.at<double>(i) = (i<=127 ? i+1 : 256-i);

  int pic_num = (int)_pics.size();
  int sam_num = (int)_points.size();
  vector<Mat> X(3, Mat(256+sam_num, 1, CV_64FC1));
  for(int i = 0; i<3; ++i) {
    Mat A = Mat::zeros(sam_num*pic_num + 257, 256 + sam_num, CV_64FC1);
    Mat B = Mat::zeros(A.rows, 1, CV_64FC1);

    int eq = 0;
    for(int j = 0; j<sam_num; ++j) for(int k = 0; k<pic_num; ++k) {
      const uchar& val = _pics[k].at<Vec3b>(_points[j])[i];
      const double& w = W.at<double>(val);
      A.at<double>(eq, val) = w;
      A.at<double>(eq, 256+j) = -w;
      B.at<double>(eq, 0) = w * log(_etimes[k]);
      ++eq;
    }
    A.at<double>(eq, 128) = 1;
    ++eq;

    for(int j = 1; j<255; ++j) {
      const double& w = W.at<double>(j);
      A.at<double>(eq, j-1) = lambda * w;
      A.at<double>(eq, j) = -2 * lambda * w;
      A.at<double>(eq, j+1) = lambda * w;
      ++eq;
    }
    solve(A, B, X[i], DECOMP_SVD);
    X[i] = X[i].rowRange(0, 256);
  }
  Size sz = _pics[0].size();
  vector<Mat> res(3);
  vector< vector<Mat> > split_pics(pic_num, vector<Mat>(3));
  for(int i = 0; i<pic_num; ++i) split(_pics[i], split_pics[i]);
  for(int c = 0; c<3; ++c) {
    Mat SUM_W(sz, CV_64FC1, Scalar::all(0)), SUM(sz, CV_64FC1, Scalar::all(0));
    for(int i = 0; i<pic_num; ++i) {
      Mat w(sz, CV_64FC1, Scalar::all(0)), val(sz, CV_64FC1, Scalar::all(0));
      LUT(split_pics[i][c], W, w);
      LUT(split_pics[i][c], X[c], val);
      SUM_W += w.mul(val - log(_etimes[i]));
      SUM += w;
    }
    exp(SUM_W.mul(1.0f / SUM), res[c]);
    res[c].convertTo(res[c], CV_32FC1);
  }
  merge(res, result);
}
