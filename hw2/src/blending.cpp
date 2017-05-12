#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "blending.hpp"
#include "projection.hpp"

pair<Point2d, Point2d> BLENDING::get_corner(const Mat& H, const Mat& src) {
  double w = src.cols, h = src.rows;
  vector<Point2d> corners = {{0, 0}, {0, h}, {w, 0}, {w, h}};
  perspectiveTransform(corners, corners, H);
  auto nx = minmax_element(corners.begin(), corners.end(),
                           [](auto& p1, auto& p2) { return p1.x < p2.x; });
  auto ny = minmax_element(corners.begin(), corners.end(),
                           [](auto& p1, auto& p2) { return p1.y < p2.y; });
  double mnx = nx.first->x, mny = ny.first->y;
  double mxx = nx.second->x, mxy = ny.second->y;
  // cerr << mnx << " " << mny << " " << mxx << " " << mxy << endl;
  return {{mnx, mny}, {mxx, mxy}};
}
void BLENDING::straightening() {
}
void BLENDING::linear() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  outputs.resize(order.size());
  for(size_t pic = 0; pic<order.size(); ++pic) {
    const auto& ord = order[pic];
    int root = ord[0].first;
    double mnx = 0, mny = 0, mxx = imgs[root].cols, mxy = imgs[root].rows;
    for(size_t i = 1; i<ord.size(); ++i) {
      int p1 = ord[i].first, p2 = ord[i].second;
      shift[p1][p1] = shift[p2][p2] * shift[p2][p1];
      Point2d pt1, pt2; tie(pt1, pt2) = get_corner(shift[p1][p1], imgs[p1]);
      mnx = min(mnx, pt1.x);
      mxx = max(mxx, pt2.x);
      mny = min(mny, pt1.y);
      mxy = max(mxy, pt2.y);
    }
    Size sz = Size(mxx-mnx, mxy-mny);
    Mat& show = outputs[pic];
    show = Mat::zeros(sz, CV_64FC3);
    Mat weight = Mat::zeros(sz, CV_64FC3);
    namedWindow("show", WINDOW_NORMAL);
    for(size_t i = 0; i<ord.size(); ++i) {
      int p = ord[i].first;
      shift[p][p].at<double>(0, 2) -= mnx;
      shift[p][p].at<double>(1, 2) -= mny;
      Mat img, dst, msk;
      imgs[p].convertTo(img, CV_64FC3);
      warpPerspective(img, dst, shift[p][p], sz);
      show += dst;
      threshold(dst, msk, 0, 1, THRESH_BINARY);
      weight += msk;

      imshow("show", msk*255);
      waitKey(0);
      Mat tmp;
      GaussianBlur(msk, tmp, Size(), 10, 10);
      imshow("show", tmp*255);
      waitKey(0);
    }
    divide(show, weight, show);
    show.convertTo(show, CV_8UC3);
  }
}
void BLENDING::average() {
  cerr << __func__;
  straightening();
  //for(auto y:order) for(auto x:y) cerr << x.first << " " << x.second << endl;
  size_t pic_num = imgs.size();
  outputs.resize(order.size());
  for(size_t pic = 0; pic<order.size(); ++pic) {
    const auto& ord = order[pic];
    int root = ord[0].first;
    double mnx = 0, mny = 0, mxx = imgs[root].cols, mxy = imgs[root].rows;
    for(size_t i = 1; i<ord.size(); ++i) {
      int p1 = ord[i].first, p2 = ord[i].second;
      shift[p1][p1] = shift[p2][p2] * shift[p2][p1];
      Point2d pt1, pt2; tie(pt1, pt2) = get_corner(shift[p1][p1], imgs[p1]);
      mnx = min(mnx, pt1.x);
      mxx = max(mxx, pt2.x);
      mny = min(mny, pt1.y);
      mxy = max(mxy, pt2.y);
    }
    //cerr << mnx << " " << mny << " " << mxx << " " << mxy << endl;
    Size sz = Size(mxx-mnx, mxy-mny);
    Mat& show = outputs[pic];
    show = Mat::zeros(sz, CV_64FC3);
    Mat weight = Mat::zeros(sz, CV_64FC3);
    for(size_t i = 0; i<ord.size(); ++i) {
      int p = ord[i].first;
      shift[p][p].at<double>(0, 2) -= mnx;
      shift[p][p].at<double>(1, 2) -= mny;
      Mat img, dst, msk;
      imgs[p].convertTo(img, CV_64FC3);
      warpPerspective(img, dst, shift[p][p], sz);
      show += dst;
      threshold(dst, msk, 0, 1, THRESH_BINARY);
      weight += msk;
    }
    divide(show, weight, show);
    show.convertTo(show, CV_8UC3);
  }
}
constexpr int SIGMA = 5;
void BLENDING::multi_band() {
  cerr << __func__;
  int pic_num = order.size();
  outputs.resize(pic_num);

  #pragma omp parallel for
  for (int pic = 0; pic < pic_num; ++pic) {
    const auto& ord = order[pic];
    int img_num = ord.size();
    int root = ord[0].first;
    double mnx = 0, mny = 0, mxx = imgs[root].cols, mxy = imgs[root].rows;
    for (int i = 1; i < img_num; ++i) {
      int p1 = ord[i].first, p2 = ord[i].second;
      shift[p1][p1] = shift[p2][p2] * shift[p2][p1];
      Point2d pt1, pt2; tie(pt1, pt2) = get_corner(shift[p1][p1], imgs[p1]);
      mnx = min(mnx, pt1.x);
      mxx = max(mxx, pt2.x);
      mny = min(mny, pt1.y);
      mxy = max(mxy, pt2.y);
    }
    Size sz = Size(mxx-mnx, mxy-mny);

    vector<vector<Mat>> weight, img, band;
    vector<Point> centers(img_num);
    weight.resize(img_num);
    img.resize(img_num);
    band.resize(img_num);
    #pragma omp parallel for
    for (int i = 0; i < img_num; ++i) {
      weight[i].resize(_para[0]);
      weight[i][0] = Mat::zeros(sz, CV_64FC1);
      img[i].resize(_para[0]);
      band[i].resize(_para[0]);
      const int& p = ord[i].first;
      shift[p][p].at<double>(0, 2) -= mnx;
      shift[p][p].at<double>(1, 2) -= mny;
      imgs[p].convertTo(img[i][0], CV_64FC3);
      Point2d pt1, pt2; tie(pt1, pt2) = get_corner(shift[p][p], imgs[p]);
      Mat mask;
      warpPerspective(img[i][0], img[i][0], shift[p][p], sz);
      threshold(img[i][0], mask, 0, 1, THRESH_BINARY);
      mask.convertTo(mask, CV_32FC3);
      cvtColor(mask, mask, COLOR_BGR2GRAY);
      int xmin = INT_MAX, xmax = 0, ymin = INT_MAX, ymax = 0;
      for (int x = 0, xn = mask.cols; x < xn; ++x)
      for (int y = 0, yn = mask.rows; y < yn; ++y) {
        if (mask.at<float>(y,x) == 0) continue;
        xmin = (x < xmin) ? x : xmin;
        xmax = (x > xmax) ? x : xmax;
        ymin = (y < ymin) ? y : ymin;
        ymax = (y > ymax) ? y : ymax;
      }
      centers[i] = {(xmin+xmax)/2, (ymin+ymax)/2};
    }
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < weight[0][0].cols; ++x)
    for (int y = 0; y < weight[0][0].rows; ++y) {
      int n, d, dmin = INT_MAX;
      for (int i = 0; i < img_num; ++i) {
        d = pow(x - centers[i].x, 2) + pow(y - centers[i].y, 2);
        if (d < dmin) { n = i; dmin = d; }
      }
      weight[n][0].at<double>(y,x) = 1;
    }
    #pragma omp parallel for
    for (int i = 0; i < img_num; ++i) {
      auto &p = img[i], &b = band[i], &w = weight[i];
      GaussianBlur(w[0], w[0], Size(0,0), SIGMA, SIGMA);
      for (int lev = 1; lev < _para[0]; ++lev) {
        GaussianBlur(p[lev-1], p[lev], Size(0,0), sqrt(2*lev-1)*SIGMA, 
                     sqrt(2*lev-1)*SIGMA);
        GaussianBlur(w[lev-1], w[lev], Size(0,0), sqrt(2*lev+1)*SIGMA,
                     sqrt(2*lev+1)*SIGMA);
        b[lev-1] = p[lev-1] - p[lev];
      }
      b[_para[0]-1] = p[_para[0]-1].clone();
    }
    Mat& show = outputs[pic];
    vector<Mat> res;
    show = Mat::zeros(sz, CV_64FC3);
    for (int b = 0; b < _para[0]; ++b) 
      res.push_back(Mat::zeros(sz, CV_64FC3));

    #pragma omp parallel for
    for (int b = 0; b < int(_para[0]); ++b) {
      Mat sum = Mat::zeros(sz, CV_64FC3);
      for (int i = 0; i < img_num; ++i) {
        Mat weight3;
        vector<Mat> weights(3, weight[i][b]);
        merge(weights, weight3);
        res[b] += band[i][b].mul(weight3);
        sum += weight3;
      }
      divide(res[b], sum, res[b]);
    }
    for (int b = 0; b < _para[0]; ++b)
      show += res[b];

    /*
    Mat& show = outputs[pic];
    show = Mat::zeros(sz, CV_64FC3);

    for (int b = 0; b < _para[0]; ++b) {
      Mat res = Mat::zeros(sz, CV_64FC1);
      Mat sum = Mat::zeros(sz, CV_64FC1);
      for (int i = 0; i < img_num; ++i) 
      for (int x = 0, xn = weight[0][0].cols; x < xn; ++x)
      for (int y = 0, yn = weight[0][0].rows; y < yn; ++y) {
        res.at<Vec3d>(y,x) += (band[i][b].at<Vec3d>(y,x)) * weight[i][b].at<double>(y,x);
        sum.at<double>(y,x) += weight[i][b].at<double>(y,x);
      }
      for (int i = 0; i < img_num; ++i) 
      for (int x = 0, xn = weight[0][0].cols; x < xn; ++x)
      for (int y = 0, yn = weight[0][0].rows; y < yn; ++y) {
        res.at<Vec3d>(y,x) /= sum.at<double>(y,x);
      }
      show += res;
    }*/

    show.convertTo(show, CV_8UC3);
  }
}
