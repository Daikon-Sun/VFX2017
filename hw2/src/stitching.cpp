#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "stitching.hpp"

bool STITCHING::is_inliner(size_t p1, size_t p2, double sx, double sy,
                           const pair<int, int>& kpid, double f = 0) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[p1][kpid1], kp2 = keypoints[p2][kpid2];
  double _sx, _sy;
  if(f > 0) {
    Size sz1 = imgs[p1].size(), sz2 = imgs[p2].size();
    double x1, y1;
    tie(x1, y1) = cylindrical_projection(f, sz1.width, sz1.height, kp1.x, kp1.y);
    double x2, y2;
    tie(x2, y2) = cylindrical_projection(f, sz2.width, sz2.height, kp2.x, kp2.y);
    _sx = x1 - x2, _sy = y1 - y2;
  } else _sx = kp1.x-kp2.x, _sy = kp1.y-kp2.y;
  return (sx-_sx) * (sx-_sx) + (sy-_sy) * (sy-_sy) < _para[1];
}
bool STITCHING::is_inliner(size_t p1, size_t p2, const Mat& sol,
                           const pair<int, int>& kpid) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[p1][kpid1], kp2 = keypoints[p2][kpid2];
  Mat pos2 = (Mat_<double>(3, 1) << kp2.x, kp2.y, 1);
  pos2 = sol * pos2;
  Mat pos1 = (Mat_<double>(2, 1) << kp1.x, kp1.y);
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
  shift.resize(pic_num, vector<Mat>(pic_num));
  #pragma omp parallel for
  for(size_t p1 = 0; p1<pic_num; ++p1)
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      int best_cnt = 0;
      double best_sx, best_sy;
      const size_t sz = match_pairs[p1][p2].size();
      assert(sz);
      for(int i = 0; i<int(_para[0]); ++i) {
        size_t id1 = rand()%sz;
        int kpid1, kpid2; tie(kpid1, kpid2) = match_pairs[p1][p2][id1];
        const auto& kp1 = keypoints[p1][kpid1], kp2 = keypoints[p2][kpid2];
        double sx = kp1.x-kp2.x, sy = kp1.y-kp2.y;
        int in_cnt = 0;
        for(size_t id3 = 0; id3<match_pairs[p1][p2].size(); ++id3)
          if(id3 != id1) 
            in_cnt += is_inliner(p1, p2, sx, sy, match_pairs[p1][p2][id3]);
        if(in_cnt > best_cnt) {
          best_cnt = in_cnt;
          best_sx = sx;
          best_sy = sy;
        }
      }
      shift[p1][p2] = (Mat_<double>(3, 3) << 1, 0, best_sx,
                                             0, 1, best_sy,
                                             0, 0,       1);
      if(!panorama_mode) break;
    }
}
void STITCHING::focal_length() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  #pragma omp parallel for
  for(size_t p1 = 0; p1<pic_num; ++p1)
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      int best_cnt1 = 0, best_cnt2 = 0;
      const size_t sz = match_pairs[p1][p2].size();
      Size sz1 = imgs[p1].size(), sz2 = imgs[p2].size();
      double best_f = 0, best_sx, best_sy;
      double w1 = sz1.width/2.0, w2 = sz2.width/2.0;
      double h1 = sz1.height/2.0, h2 = sz2.height/2.0;
      for(int i = 0; i<int(_para[0]); ++i) {
        size_t id1 = rand()%sz;
        size_t id2 = (id1+(rand()%(sz-1))+1)%sz;
        int kpid11, kpid21; tie(kpid11, kpid21) = match_pairs[p1][p2][id1];
        int kpid12, kpid22; tie(kpid12, kpid22) = match_pairs[p1][p2][id2];
        const auto& kp11 = keypoints[p1][kpid11];
        const auto& kp21 = keypoints[p2][kpid21];
        const auto& kp12 = keypoints[p1][kpid12];
        const auto& kp22 = keypoints[p2][kpid22];
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
        for(size_t id3 = 0; id3<match_pairs[p1][p2].size(); ++id3)
          if(id3 != id1) {
            in_cnt1 += is_inliner(p1, p2, sx, sy, match_pairs[p1][p2][id3]);
            _para[0] /= 2;
            in_cnt2 += is_inliner(p1, p2, sx, sy, match_pairs[p1][p2][id3]);
            _para[0] *= 2;
          }
        if(in_cnt1>best_cnt1 || (in_cnt1==best_cnt1 && in_cnt2<best_cnt2)) {
          best_cnt1 = in_cnt1;
          best_cnt2 = in_cnt2;
          best_sx = sx;
          best_sy = sy;
          best_f = f;
        }
      }
      shift[p1][p2] = (Mat_<double>(3, 3) << 1, best_f, best_sx,
                                             0,      1, best_sy,
                                             0,      0,       1);
      if(!panorama_mode) break;
    }
}
void STITCHING::rotation() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  #pragma omp parallel for
  for(size_t p1 = 0; p1<pic_num; ++p1)
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      Mat best_sol;
      int best_cnt1 = 0, best_cnt2 = 0;
      const size_t sz = match_pairs[p1][p2].size();
      for(int i = 0; i<int(_para[0]); ++i) {
        size_t id1 = rand()%sz;
        size_t id2 = (id1+(rand()%(sz-1))+1)%sz;
        size_t id3 = rand()%sz;
        while(id3 == id1 || id3 == id2) id3 = rand()%sz;
        int kpid11, kpid12; tie(kpid11, kpid12) = match_pairs[p1][p2][id1];
        int kpid21, kpid22; tie(kpid21, kpid22) = match_pairs[p1][p2][id2];
        int kpid31, kpid32; tie(kpid31, kpid32) = match_pairs[p1][p2][id3];
        const auto& kp11 = keypoints[p1][kpid11];
        const auto& kp21 = keypoints[p1][kpid21];
        const auto& kp31 = keypoints[p1][kpid31];
        const auto& kp12 = keypoints[p2][kpid12];
        const auto& kp22 = keypoints[p2][kpid22];
        const auto& kp32 = keypoints[p2][kpid32];
        Mat rot = (Mat_<double>(3, 3) << kp12.x, kp22.x, kp32.x,
                                         kp12.y, kp22.y, kp32.y,
                                            1.0,    1.0,    1.0);
        Mat pos = (Mat_<double>(2, 3) << kp11.x, kp21.x, kp31.x,
                                         kp11.y, kp21.y, kp31.y);
        Mat sol = pos * rot.inv();
        int in_cnt1 = 0, in_cnt2 = 0;
        for(size_t id3 = 0; id3<match_pairs[p1][p2].size(); ++id3)
          if(id3 != id1) {
            in_cnt1 += is_inliner(p1, p2, sol, match_pairs[p1][p2][id3]);
            _para[1] /= 2;
            in_cnt2 += is_inliner(p1, p2, sol, match_pairs[p1][p2][id3]);
            _para[1] *= 2;
          }
        if(in_cnt1>best_cnt1 || (in_cnt1==best_cnt1 && in_cnt2>best_cnt2)) {
          best_cnt1 = in_cnt1;
          best_cnt2 = in_cnt2;
          sol.copyTo(best_sol);
        }
      }
      copyMakeBorder(best_sol, best_sol, 0, 1, 0, 0, BORDER_CONSTANT, 0);
      best_sol.at<double>(2, 2) = 1;
      shift[p1][p2] = best_sol.clone();
      if(!panorama_mode) break;
    }
}
constexpr int M = 1;
void STITCHING::autostitch() {
  cerr << __func__; 
  size_t pic_num = imgs.size();
  //vector<vector<Mat>> msk(pic_num, vector<Mat>(pic_num));
  vector<vector<pair<int, int>>> in_cnt(pic_num, vector<pair<int,int>>(pic_num));
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  
  for(size_t p1 = 0; p1<pic_num; ++p1) {
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      vector<Point2f> src, dst;      
      for(const auto& mp : match_pairs[p1][p2]) {
        int k1, k2; tie(k1, k2) = mp;
        const auto& kp1 = keypoints[p1][k1];
        const auto& kp2 = keypoints[p2][k2];
        src.emplace_back(kp2.x, kp2.y);
        dst.emplace_back(kp1.x, kp1.y);
      }
      Mat msk;
      shift[p1][p2] = findHomography(src, dst, CV_RANSAC, _para[1], msk);
      int sm = sum(msk)[0];
      in_cnt[p1][p2] = {sm, p2};
      in_cnt[p2][p1] = {sm, p1};
      if(!panorama_mode) break;
    }
  }
  vector<pair<int,int>> connect;
  for(size_t i = 0; i<pic_num; ++i) {
    sort(in_cnt[i].begin(), in_cnt[i].end(), greater<pair<int,int>>()); 
    for(size_t j = 0; j<M && j<in_cnt[i].size(); ++j) {
      int p1 = i, p2 = in_cnt[i][j].second;
      if(p1 >= p2) swap(p1, p2);
      if(in_cnt[i][j].first > 5.9+0.22*match_pairs[p1][p2].size())
        connect.emplace_back(p1, p2);
    }
  }
  sort(connect.begin(), connect.end());
  connect.resize(unique(connect.begin(), connect.end()) - connect.begin());
}
