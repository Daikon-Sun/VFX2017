#include "hdr.hpp"

void generate_points(const Mat& m, vector<Point>& _points) {
  constexpr int sam_num = 50, range = 3;
  _points.reserve(sam_num);
  while(_points.size() < sam_num) {
    const int r = rand()%(m.rows-10)+5;
    const int c = rand()%(m.cols-10)+5;
    const Vec3b& val = m.at<Vec3b>(r, c);
    bool same = true;
    for(int j = r-range; j<=r+range; ++j) for(int k = c-range; k<=c+range; ++k) {
      if(m.at<Vec3b>(j, k) != val) {
        same = false;
        break;
      }
    }
    if( same ) _points.emplace_back(c, r);
  }
}

void DEBEVEC::process(Mat& result, double lambda) {

  Mat W(1, 256, CV_64FC1);
  //for(int i = 0; i<256; ++i) W.at<double>(i) = 1-pow(2.0*i/255-1, 20);
  for(int i = 0; i<256; ++i) W.at<double>(i) = (i<=127 ? i+1 : 256-i);
  
  vector<Point> _points;
  generate_points(_pics[0], _points);
  
  int pic_num = (int)_pics.size();
  int sam_num = (int)_points.size();
  vector<Mat> X(3);
  for(int c = 0; c<3; ++c) {
    X[c] = Mat(256+sam_num, 1, CV_64FC1);
    Mat A = Mat::zeros(sam_num*pic_num + 257, 256 + sam_num, CV_64FC1);
    Mat B = Mat::zeros(A.rows, 1, CV_64FC1);

    int eq = 0;
    for(int j = 0; j<sam_num; ++j) for(int k = 0; k<pic_num; ++k) {
      const uchar& val = _pics[k].at<Vec3b>(_points[j])[c];
      CV_Assert(val>=0 && val<=255);
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
    solve(A, B, X[c], DECOMP_SVD);
    X[c] = X[c].rowRange(0, 256);
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

void MERTENS::process(Mat& result) {
  int pic_num = (int)_pics.size();
  Size sz = _pics[0].size();
  vector<Mat> W(pic_num);
  Mat ALL_W(sz, CV_64FC1, Scalar::all(0)); 
  vector<Mat> pics(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    Mat img, gray, C, S = Mat(sz, CV_64FC1, Scalar::all(0));
    Mat E = Mat(sz, CV_64FC1, Scalar::all(1)), diff, s;
    _pics[i].convertTo(pics[i], CV_64FC3, 1.0/255);
    vector<Mat> split_pic(3);
    split(pics[i], split_pic);
    pics[i].convertTo(img, CV_32FC3);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_64FC1);
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
    constexpr double wc = 1, ws = 1, we = 1;
    pow(C, wc, C);
    pow(S, ws, S);
    pow(E, we, E);
    W[i] = W[i].mul(C);
    W[i] = W[i].mul(S);
    W[i] = W[i].mul(E)+1e-9;
    ALL_W += W[i];
  }
  Mat up;
  constexpr int max_lev = 7;
  vector<Mat> final_pics(max_lev + 1);
  vector< vector<Mat> > pics_pyr(pic_num), W_pyr(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    W[i] /= ALL_W;
    buildPyramid(pics[i], pics_pyr[i], max_lev);
    buildPyramid(W[i], W_pyr[i], max_lev);
  }
  for(int i = 0; i<=max_lev; ++i)
    final_pics[i] = Mat(pics_pyr[0][i].size(), CV_64FC3, Scalar::all(0));
  for(int i = 0; i<pic_num; ++i) {
    for(int j = 0; j<max_lev; ++j) {
      pyrUp(pics_pyr[i][j+1], up,  pics_pyr[i][j].size());
      pics_pyr[i][j] -= up;
    }
    for(int j = 0; j<=max_lev; ++j) {
      vector<Mat> split_pics_pyr(3);
      split(pics_pyr[i][j], split_pics_pyr);
      for(int c = 0; c<3; ++c) 
        split_pics_pyr[c] = split_pics_pyr[c].mul(W_pyr[i][j]);
      merge(split_pics_pyr, pics_pyr[i][j]);
      final_pics[j] += pics_pyr[i][j];
    }
  }
  for(int i = max_lev-1; i>=0; --i) {
    pyrUp(final_pics[i+1], up, final_pics[i].size());
    final_pics[i] += up;
  }
  result = final_pics[0].clone();
}
