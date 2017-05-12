#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "projection.hpp"

void PROJECTION::set_focal_length(double f) {
  if(_para.size() < 1) _para.push_back(f);
  else _para[0] = f;
}
pair<double, double> PROJECTION::projected_xy(size_t i, double w, double h, 
                                              double x, double y) {
  Point2d p1 = {_para[0]*atan((x-w/2)/_para[0]) + w/2, 
                _para[0]*(y-h/2)/sqrt(_para[0]*_para[0]+(x-w/2)*(x-w/2)) + h/2};
  double&& para = (double)imgs[i].rows / imgs[i].cols * _para[0];
  return {para*(p1.x-w/2)/sqrt(para*para+(p1.y-h/2)*(p1.y-h/2))+w/2,
          para*atan((p1.y-h/2)/para) + h/2};
}
void PROJECTION::no_projection() {
  cerr <<__func__;
}
void PROJECTION::cylindrical() {
  cerr <<__func__;
  Size sz = imgs[0].size();
  #pragma omp parallel for
  for(size_t i = 0; i<imgs.size(); ++i) {
    Mat img = Mat(sz, CV_8UC3, Scalar(0, 0, 0));
    double mxx = 0, mxy = 0, mnx = DBL_MAX, mny = DBL_MAX;
    for(int y = 0; y<sz.height; ++y) for(int x = 0; x<sz.width; ++x) {
      double nx, ny; tie(nx, ny) = projected_xy(i, sz.width, sz.height, x, y);
      img.at<Vec3b>(ny, nx) = imgs[i].at<Vec3b>(y, x); 
      mxx = max(mxx, nx);
      mxy = max(mxy, ny);
      mnx = min(mnx, nx);
      mny = min(mny, ny);
    }
    img(Rect(mnx, mny, mxx-mnx, mxy-mny)).copyTo(imgs[i]);
  }
  #pragma omp parallel for
  for(size_t i = 0; i<keypoints.size(); ++i)
    for(size_t j = 0; j<keypoints[i].size(); ++j)
      tie(keypoints[i][j].x, keypoints[i][j].y) = 
        projected_xy(i, sz.width, sz.height, 
                     keypoints[i][j].x, keypoints[i][j].y);
}
