#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "hdr.hpp"

void DEBEVEC::process(const vector<Mat>& pics, const vector<double>& etimes,
                      const vector<Mat>& gW, Mat& result) {
  const double lambda = _para[0];
  Mat W(1, 256, CV_64FC1);
  for(int i = 0; i<256; ++i) W.at<double>(i) = (i<=127 ? i+1 : 256-i);
  
  vector<Point> points;
  generate_points(pics[0], _para[1], points);
  
  int pic_num = (int)pics.size();
  int sam_num = (int)points.size();
  vector<Mat> X(3);
  for(int c = 0; c<3; ++c) {
    X[c] = Mat(256+sam_num, 1, CV_64FC1);
    Mat A = Mat::zeros(sam_num*pic_num + 257, 256 + sam_num, CV_64FC1);
    Mat B = Mat::zeros(A.rows, 1, CV_64FC1);

    int eq = 0;
    for(int j = 0; j<sam_num; ++j) for(int k = 0; k<pic_num; ++k) {
      const uchar& val = pics[k].at<Vec3b>(points[j])[c];
      CV_Assert(val>=0 && val<=255);
      const double& w = W.at<double>(val);
      A.at<double>(eq, val) = w;
      A.at<double>(eq, 256+j) = -w;
      B.at<double>(eq, 0) = w * log(etimes[k]);
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
    solve(A, B, X[c], DECOMP_SVD);
    X[c] = X[c].rowRange(0, 256);
  }
  Size sz = pics[0].size();
  vector<Mat> res(3);
  vector< vector<Mat> > split_pics(pic_num, vector<Mat>(3));
  for(int i = 0; i<pic_num; ++i) split(pics[i], split_pics[i]);
  for(int c = 0; c<3; ++c) {
    Mat SUM_W(sz, CV_64FC1, Scalar::all(0)), SUM(sz, CV_64FC1, Scalar::all(0));
    for(int i = 0; i<pic_num; ++i) {
      Mat w(sz, CV_64FC1, Scalar::all(0)), val(sz, CV_64FC1, Scalar::all(0));
      if(gW.empty()) {
        LUT(split_pics[i][c], W, w);
        LUT(split_pics[i][c], X[c], val);
        SUM_W += w.mul(val - log(etimes[i]));
        SUM += w;
      } else {
        LUT(split_pics[i][c], X[c], val);
        SUM_W += gW[i].mul(val - log(etimes[i]));
        SUM += gW[i];
      }
    }
    exp(SUM_W.mul(1.0f / SUM), res[c]);
    res[c].convertTo(res[c], CV_32FC1);
  }
  merge(res, result);
}

void MERTENS::process(const vector<Mat>& pics,
                      const vector<Mat>& gW, Mat& result) {
  int pic_num = (int)pics.size();
  Size sz = pics[0].size();
  vector<Mat> W(pic_num);
  Mat ALL_W(sz, CV_64FC1, Scalar::all(0)); 
  vector<Mat> _pics(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    Mat gray, C, S = Mat(sz, CV_64FC1, Scalar::all(0));
    Mat E = Mat(sz, CV_64FC1, Scalar::all(1)), diff, s;
    pics[i].convertTo(_pics[i], CV_64FC3, 1.0/255);
    vector<Mat> split_pic(3);
    split(_pics[i], split_pic);
    mycvtColor(_pics[i], gray);
    //contrast
    Laplacian(gray, C, CV_64FC1);
    C = abs(C);
    //saturation
    Mat mean = (split_pic[0]+split_pic[1]+split_pic[2]) / 3.0;
    for(int c = 0; c<3; ++c) {
      pow(split_pic[c]-mean, 2, diff);
      S += diff;
    }
    sqrt(S/3.0, S);
    //well-exposured
    for(int c = 0; c<3; ++c) {
      pow((split_pic[c]-0.5) / (0.2 * sqrt(2)), 2, s);
      exp(-s, s);
      E = E.mul(s);
    } 
    W[i] = Mat(sz, CV_64FC1, Scalar::all(1));
    pow(C, _para[0], C);
    pow(S, _para[1], S);
    pow(E, _para[2], E);
    W[i] = W[i].mul(C);
    W[i] = W[i].mul(S);
    if(!gW.empty()) {
      exp(gW[i], gW[i]);
      W[i] = W[i].mul(gW[i]);
    }
    W[i] = W[i].mul(E)+1e-20;
    ALL_W += W[i];
  }
  Mat up;
  vector<Mat> final_pics(_para[3] + 1);
  vector< vector<Mat> > pics_pyr(pic_num), W_pyr(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    W[i] /= ALL_W;
    buildPyramid(_pics[i], pics_pyr[i], _para[3]);
    buildPyramid(W[i], W_pyr[i], _para[3]);
  }
  for(int i = 0; i<=_para[3]; ++i)
    final_pics[i] = Mat(pics_pyr[0][i].size(), CV_64FC3, Scalar::all(0));
  for(int i = 0; i<pic_num; ++i) {
    for(int j = 0; j<_para[3]; ++j) {
      pyrUp(pics_pyr[i][j+1], up,  pics_pyr[i][j].size());
      pics_pyr[i][j] -= up;
    }
    for(int j = 0; j<=_para[3]; ++j) {
      vector<Mat> split_pics_pyr(3);
      split(pics_pyr[i][j], split_pics_pyr);
      for(int c = 0; c<3; ++c) 
        split_pics_pyr[c] = split_pics_pyr[c].mul(W_pyr[i][j]);
      merge(split_pics_pyr, pics_pyr[i][j]);
      final_pics[j] += pics_pyr[i][j];
    }
  }
  for(int i = _para[3]-1; i>=0; --i) {
    pyrUp(final_pics[i+1], up, final_pics[i].size());
    final_pics[i] += up;
  }
  result = final_pics[0].clone();
}
