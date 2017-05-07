#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "stitching.hpp"

bool STITCHING::is_inliner(size_t pic, double sx, double sy,
                           const pair<int, int>& kpid, double f = 0) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[pic][kpid1], kp2 = keypoints[pic+1][kpid2];
  double _sx, _sy;
  if(f > 0) {
    Size sz1 = imgs[pic].size(), sz2 = imgs[pic+1].size();
    double x1, y1;
    tie(x1, y1) = cylindrical_projection(f, sz1.width, sz1.height, kp1.x, kp1.y);
    double x2, y2;
    tie(x2, y2) = cylindrical_projection(f, sz2.width, sz2.height, kp2.x, kp2.y);
    _sx = x1 - x2, _sy = y1 - y2;
  } else _sx = kp1.x-kp2.x, _sy = kp1.y-kp2.y;
  return (sx-_sx) * (sx-_sx) + (sy-_sy) * (sy-_sy) < _para[1];
}
bool STITCHING::is_inliner(size_t pic, const Mat& sol,
                           const pair<int, int>& kpid) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[pic][kpid1], kp2 = keypoints[pic+1][kpid2];
  Mat pos1 = (Mat_<double>(3, 1) << kp1.x, kp1.y, 1);
  pos1 = sol * pos1;
  Mat pos2 = (Mat_<double>(2, 1) << kp2.x, kp2.y);
  Mat err = pos1 - pos2;
  return sum(err.mul(err))[0] < _para[1];
}
pair<double, double>
STITCHING::cylindrical_projection(double f, double w, double h, 
                                  double x, double y) {
  return {f*atan((x-w/2)/f) + w/2, f*(y-h/2)/sqrt(f*f+(x-w/2)*(x-w/2)) + h/2};
}
void STITCHING::translation() { 
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num-1);
  //#pragma omp parallel for
  for(size_t pic = 0; pic<match_pairs.size(); ++pic) {
    int best_cnt = 0, best_sx, best_sy;
    const size_t sz = match_pairs[pic].size();
    for(int i = 0; i<int(_para[0]); ++i) {
      size_t id1 = rand()%sz;
      int kpid1, kpid2; tie(kpid1, kpid2) = match_pairs[pic][id1];
      const auto& kp1 = keypoints[pic][kpid1], kp2 = keypoints[pic+1][kpid2];
      double sx = kp1.x-kp2.x, sy = kp1.y-kp2.y;
      int in_cnt = 0;
      for(size_t id3 = 0; id3<match_pairs[pic].size(); ++id3) if(id3 != id1)
        in_cnt += is_inliner(pic, sx, sy, match_pairs[pic][id3]);
      if(in_cnt > best_cnt) {
        best_cnt = in_cnt;
        cerr << best_cnt << endl;
        best_sx = sx;
        best_sy = sy;
      }
    }
    shift[pic] = {best_sx, best_sy};
    cerr << best_sy << " " << best_sy << endl;
    Size sz1 = imgs[pic].size();
    Size sz2 = imgs[pic+1].size();
    const Mat& img0 = imgs[pic];
    const Mat& img1 = imgs[pic+1];
    Mat show = Mat::zeros(sz1.height+abs(best_sy), sz2.width+abs(best_sx), 
                          CV_8UC3);
    Mat left(show, Rect(0, max(0, -best_sy), sz1.width, sz1.height));
    Mat right(show, Rect(best_sx, max(0, best_sy), sz2.width, sz2.height));
    img0.copyTo(left);
    img1.copyTo(right);
    namedWindow("translation", WINDOW_NORMAL);
    imshow("translation", show);
    waitKey(0);
  }
}
void STITCHING::focal_length() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num-1);
  #pragma omp parallel for
  for(size_t pic = 0; pic<match_pairs.size(); ++pic) {
    int best_cnt1 = 0, best_cnt2 = 0, best_sx, best_sy;
    const size_t sz = match_pairs[pic].size();
    Size sz1 = imgs[pic].size(), sz2 = imgs[pic+1].size();
    double best_f = 0;
    double w1 = sz1.width/2.0, w2 = sz2.width/2.0;
    double h1 = sz1.height/2.0, h2 = sz2.height/2.0;
    for(int i = 0; i<int(_para[0]); ++i) {
      size_t id1 = rand()%sz;
      size_t id2 = (id1+(rand()%(sz-1))+1)%sz;
      int kpid11, kpid21; tie(kpid11, kpid21) = match_pairs[pic][id1];
      int kpid12, kpid22; tie(kpid12, kpid22) = match_pairs[pic][id2];
      const auto& kp11 = keypoints[pic][kpid11];
      const auto& kp21 = keypoints[pic+1][kpid21];
      const auto& kp12 = keypoints[pic][kpid12];
      const auto& kp22 = keypoints[pic+1][kpid22];
      double nu = (kp21.x-w2) * (kp22.x-w2) * (kp12.x-kp11.x)
                -(kp11.x-w1) * (kp12.x-w1) * (kp22.x-kp21.x);
      double de = (kp22.x-kp21.x + kp11.x-kp12.x);
      if(abs(de) >= 25 || nu*de < 0) continue;
      double f = sqrt(nu / de);
      double nx11, ny11; 
      tie(nx11, ny11) = cylindrical_projection(f, w1*2, h1*2, kp11.x, kp11.y);
      double nx12, ny12; 
      tie(nx12, ny12) = cylindrical_projection(f, w1*2, h1*2, kp12.x, kp12.y);
      double nx21, ny21; 
      tie(nx21, ny21) = cylindrical_projection(f, w2*2, h2*2, kp21.x, kp21.y);
      double nx22, ny22;
      tie(nx22, ny22) = cylindrical_projection(f, w2*2, h2*2, kp22.x, kp22.y);
      double sx1 = (nx11-nx21), sx2 = (nx12-nx22);
      double sy1 = (ny11-ny21), sy2 = (ny12-ny22);
      if(abs(sx1-sx2) > 5 || abs(sy1-sy2) > 5) continue;
      double sx = (sx1+sx2)/2, sy = (sy1+sy2)/2;
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
        best_sx = sx;
        best_sy = sy;
        best_f = f;
      }
    }
    shift[pic] = {best_sx, best_sy};
  }
}
void STITCHING::rotation() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num-1);
  #pragma omp parallel for
  for(size_t pic = 0; pic<match_pairs.size(); ++pic) {
    Mat best_sol;
    int best_cnt1 = 0, best_cnt2 = 0;
    const size_t sz = match_pairs[pic].size();
    for(int i = 0; i<int(_para[0]); ++i) {
      size_t id1 = rand()%sz;
      size_t id2 = (id1+(rand()%(sz-1))+1)%sz;
      size_t id3 = rand()%sz;
      while(id3 == id1 || id3 == id2) id3 = rand()%sz;
      int kpid11, kpid12; tie(kpid11, kpid12) = match_pairs[pic][id1];
      int kpid21, kpid22; tie(kpid21, kpid22) = match_pairs[pic][id2];
      int kpid31, kpid32; tie(kpid31, kpid32) = match_pairs[pic][id3];
      const auto& kp11 = keypoints[pic][kpid11];
      const auto& kp21 = keypoints[pic][kpid21];
      const auto& kp31 = keypoints[pic][kpid31];
      const auto& kp12 = keypoints[pic+1][kpid12];
      const auto& kp22 = keypoints[pic+1][kpid22];
      const auto& kp32 = keypoints[pic+1][kpid32];
      Mat rot = (Mat_<double>(3, 3) << kp11.x, kp21.x, kp31.x,
                                      kp11.y, kp21.y, kp31.y,
                                         1.0,    1.0,    1.0);
      Mat pos = (Mat_<double>(2, 3) << kp12.x, kp22.x, kp32.x,
                                      kp12.y, kp22.y, kp32.y);
      Mat sol = pos * rot.inv();
      int in_cnt1 = 0, in_cnt2 = 0;
      for(size_t id3 = 0; id3<match_pairs[pic].size(); ++id3) if(id3 != id1) {
        in_cnt1 += is_inliner(pic, sol, match_pairs[pic][id3]);
        _para[1] /= 2;
        in_cnt2 += is_inliner(pic, sol, match_pairs[pic][id3]);
        _para[1] *= 2;
      }
      if(in_cnt1 > best_cnt1 || (in_cnt1 == best_cnt1 && in_cnt2 > best_cnt2)) {
        best_cnt1 = in_cnt1;
        best_cnt2 = in_cnt2;
        sol.copyTo(best_sol);
      }
    }
    Size sz1 = imgs[pic].size();
    Size sz2 = imgs[pic+1].size();
    double minx = 0, miny = 0, maxx = sz2.width, maxy = sz2.height;
    for(int x = 0; x<sz1.width; ++x) for(int y = 0; y<sz1.height; ++y) {
      Mat pos = (Mat_<double>(3, 1) << x, y, 1);
      pos = best_sol * pos;
      minx = min(minx, pos.at<double>(0, 0));
      maxx = max(maxx, pos.at<double>(0, 0));
      miny = min(miny, pos.at<double>(1, 0));
      maxy = max(maxy, pos.at<double>(1, 0));
    }
    const Mat& img1 = imgs[pic];
    const Mat& img2 = imgs[pic+1];
    Mat show = Mat::zeros(int(maxy-miny)+1, int(maxx-minx)+1, CV_8UC3);
    for(int x = 0; x<sz1.width; ++x) for(int y = 0; y<sz1.height; ++y) {
      Mat pos = (Mat_<double>(3, 1) << x, y, 1);
      pos = best_sol * pos;
      int nx = int(pos.at<double>(0, 0)+0.5-minx);
      int ny = int(pos.at<double>(1, 0)+0.5-miny);
      show.at<Vec3b>(ny, nx) = img1.at<Vec3b>(y, x);
    }
    Mat right(show, Rect(-minx, -miny, sz2.width, sz2.height));
    img2.copyTo(right);
    namedWindow("rotation", WINDOW_NORMAL);
    imshow("rotation", show);
    waitKey(0);
  }
}
