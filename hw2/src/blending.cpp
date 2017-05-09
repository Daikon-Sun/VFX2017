#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "blending.hpp"

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
  //cerr << mnx << " " << mny << " " << mxx << " " << mxy << endl;
  return {{mnx, mny}, {mxx, mxy}};
}
void BLENDING::linear() {
  cerr << __func__;
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
