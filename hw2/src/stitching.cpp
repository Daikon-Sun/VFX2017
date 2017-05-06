#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "stitching.hpp"

bool STITCHING::is_inliner(size_t pic, float sx, float sy,
                           const pair<int, int>& kpid, float f = 0) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[pic][kpid1], kp2 = keypoints[pic+1][kpid2];
  float _sx, _sy;
  if(f > 0) {
    Size sz1 = imgs[pic].size(), sz2 = imgs[pic+1].size();
    float x1, y1;
    tie(x1, y1) = cylindrical_projection(f, sz1.width, sz1.height, kp1.x, kp1.y);
    float x2, y2;
    tie(x2, y2) = cylindrical_projection(f, sz2.width, sz2.height, kp2.x, kp2.y);
    _sx = x1 - x2, _sy = y1 - y2;
  } else _sx = kp1.x-kp2.x, _sy = kp1.y-kp2.y;
  return (sx-_sx) * (sx-_sx) + (sy-_sy) * (sy-_sy) < _para[1];
}
void STITCHING::translation() { 
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num-1);
  #pragma omp parallel for
  for(size_t pic = 0; pic<match_pairs.size(); ++pic) {
    int best_cnt = 0, best_pair, best_sx, best_sy;
    const size_t sz = match_pairs[pic].size();
    Size sz1 = imgs[pic].size(), sz2 = imgs[pic+1].size();
    float w1 = sz1.width/2.0, w2 = sz2.width/2.0;
    float h1 = sz1.height/2.0, h2 = sz2.height/2.0;
    for(int i = 0; i<int(_para[0]); ++i) {
      size_t id1 = rand()%sz;
      int kpid11, kpid21; tie(kpid11, kpid21) = match_pairs[pic][id1];
      const auto& kp11 = keypoints[pic][kpid11], kp21 = keypoints[pic+1][kpid21];
      float sx = kp11.x-kp21.x, sy = kp11.y-kp21.y;
      int in_cnt = 0;
      for(size_t id3 = 0; id3<match_pairs[pic].size(); ++id3) if(id3 != id1)
        in_cnt += is_inliner(pic, sx, sy, match_pairs[pic][id3]);
      if(in_cnt > best_cnt) {
        best_cnt = in_cnt;
        best_pair = id1;
        best_sx = sx;
        best_sy = sy;
      }
    }
    shift[pic] = {best_sx, best_sy};
    sz1 = imgs[pic].size();
    sz2 = imgs[pic+1].size();
    const Mat& img0 = imgs[pic];
    const Mat& img1 = imgs[pic+1];
    Mat show = Mat::zeros(sz1.height+abs(best_sy), sz2.width+abs(best_sx), 
                          CV_8UC3);
    Mat left(show, Rect(0, max(0, -best_sy), sz1.width, sz1.height));
    Mat right(show, Rect(best_sx, max(0, best_sy), sz2.width, sz2.height));
    img0.copyTo(left);
    img1.copyTo(right);
    namedWindow("process", WINDOW_NORMAL);
    imshow("process", show);
    waitKey(0);
  }
}
pair<float, float> STITCHING::cylindrical_projection(float f, float w, float h, 
                                                     float x, float y) {
  return {f*atanf((x-w/2)/f) + w/2, f*(y-h/2)/sqrtf(f*f+(x-w/2)*(x-w/2)) + h/2};
}
void STITCHING::focal_length() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num-1);
  #pragma omp parallel for
  for(size_t pic = 0; pic<match_pairs.size(); ++pic) {
    int best_cnt1 = 0, best_cnt2 = 0, best_pair, best_sx, best_sy;
    const size_t sz = match_pairs[pic].size();
    Size sz1 = imgs[pic].size(), sz2 = imgs[pic+1].size();
    float best_f = 0;
    float w1 = sz1.width/2.0, w2 = sz2.width/2.0;
    float h1 = sz1.height/2.0, h2 = sz2.height/2.0;
    for(int i = 0; i<int(_para[0]); ++i) {
      size_t id1 = rand()%sz;
      size_t id2 = (id1+(rand()%(sz-1))+1)%sz;
      int kpid11, kpid21; tie(kpid11, kpid21) = match_pairs[pic][id1];
      int kpid12, kpid22; tie(kpid12, kpid22) = match_pairs[pic][id2];
      const auto& kp11 = keypoints[pic][kpid11];
      const auto& kp21 = keypoints[pic+1][kpid21];
      const auto& kp12 = keypoints[pic][kpid12];
      const auto& kp22 = keypoints[pic+1][kpid22];
      float nu = (kp21.x-w2) * (kp22.x-w2) * (kp12.x-kp11.x)
                -(kp11.x-w1) * (kp12.x-w1) * (kp22.x-kp21.x);
      float de = (kp22.x-kp21.x + kp11.x-kp12.x);
      if(fabs(de) >= 25 || nu*de < 0) continue;
      float f = sqrtf(nu / de);
      float nx11, ny11; 
      tie(nx11, ny11) = cylindrical_projection(f, w1*2, h1*2, kp11.x, kp11.y);
      float nx12, ny12; 
      tie(nx12, ny12) = cylindrical_projection(f, w1*2, h1*2, kp12.x, kp12.y);
      float nx21, ny21; 
      tie(nx21, ny21) = cylindrical_projection(f, w2*2, h2*2, kp21.x, kp21.y);
      float nx22, ny22;
      tie(nx22, ny22) = cylindrical_projection(f, w2*2, h2*2, kp22.x, kp22.y);
      float sx1 = (nx11-nx21), sx2 = (nx12-nx22);
      float sy1 = (ny11-ny21), sy2 = (ny12-ny22);
      if(fabs(sx1-sx2) > 5 || fabs(sy1-sy2) > 5) continue;
      float sx = (sx1+sx2)/2, sy = (sy1+sy2)/2;
      int in_cnt1 = 0, in_cnt2 = 0;
      for(size_t id3 = 0; id3<match_pairs[pic].size(); ++id3) if(id3 != id1) {
        in_cnt1 += is_inliner(pic, sx, sy, match_pairs[pic][id3]);
        _para[0] /= 2;
        in_cnt2 += is_inliner(pic, sx, sy, match_pairs[pic][id3]);
        _para[0] *= 2;
      }
      if(in_cnt1 > best_cnt1 || (in_cnt1 == best_cnt1 && in_cnt2 < best_cnt2)) {
        best_cnt1 = in_cnt1;
        best_cnt2 = in_cnt2;
        best_pair = id1;
        best_sx = sx;
        best_sy = sy;
        best_f = f;
      }
    }
    shift[pic] = {best_sx, best_sy};
  }
}
