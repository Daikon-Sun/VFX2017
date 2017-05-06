#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "projection.hpp"

pair<float, float> PROJECTION::projected_xy(float w, float h, float x, float y) {
  return {_para[0]*atanf((x-w/2)/_para[0]) + w/2,
          _para[0]*(y-h/2)/sqrtf(_para[0]*_para[0]+(x-w/2)*(x-w/2)) + h/2};
}
void PROJECTION::no_projection() {
  cerr <<__func__;
}
void PROJECTION::cylindrical() {
  cerr <<__func__;
  Size sz = imgs[0].size();
  for(size_t i = 0; i<imgs.size(); ++i) {
    Mat img = Mat(sz, CV_8UC3, Scalar(0, 0, 0));
    int mxx = 0, mxy = 0, mnx = INT_MAX, mny = INT_MAX;
    for(int y = 0; y<sz.height; ++y) for(int x = 0; x<sz.width; ++x) {
      int nx, ny; tie(nx, ny) = projected_xy(sz.width, sz.height, x, y);
      img.at<Vec3b>(ny, nx) = imgs[i].at<Vec3b>(y, x); 
      mxx = max(mxx, nx);
      mxy = max(mxy, ny);
      mnx = min(mnx, nx);
      mny = min(mny, ny);
    }
    img(Rect(mnx, mny, mxx-mnx+1, mxy-mny+1)).copyTo(imgs[i]);
  }
  for(size_t i = 0; i<keypoints.size(); ++i)
    for(size_t j = 0; j<keypoints[i].size(); ++j)
      tie(keypoints[i][j].x, keypoints[i][j].y) = 
        projected_xy(sz.width, sz.height, keypoints[i][j].x, keypoints[i][j].y);
}
