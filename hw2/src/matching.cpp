#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "matching.hpp"

MATCHING::MATCHING(const int& p_mode,
                   const vector<double>& para,
                   const vector<Mat>& i,
                   vector<vector<Keypoint> >& k,
                   vector<vector<vector<pair<int,int>>>>& m)
                  : panorama_mode(p_mode), _para(para), imgs(i), 
                    keypoints(k), match_pairs(m) {};

void MATCHING::HAAR() {
  cerr << __func__;
  int BIN_NUM = _para[0];
  double BOUND = BIN_NUM/2.0-0.5;
  double THRESHOLD = _para[1];
  Point pts[3] = {Point(1, 0), Point(0, 1), Point(1, 1)};

  size_t tot_kpts = 0, pic_num = keypoints.size();
  for(size_t i = 0; i<keypoints.size(); ++i) tot_kpts += keypoints[i].size();

  double mean[3], sd[3];
  #pragma omp parallel for
  for(int i = 0; i<3; ++i) {
    vector<double> v(tot_kpts), diff(tot_kpts);
    size_t cnt = 0;
    for(size_t pic = 0; pic<pic_num; ++pic)
      for(const auto& p:keypoints[pic])
        v[cnt++] = p.patch.at<double>(pts[i]);
    mean[i] = std::accumulate(v.begin(), v.end(), 0.0) / tot_kpts;
    transform(v.begin(), v.end(), diff.begin(), 
             [mean, i](const double& v) { return v-mean[i]; });
    sd[i] = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    sd[i] = sqrt(sd[i] / diff.size()) * 6.0 / BIN_NUM;
  }

  list<int> table[pic_num][BIN_NUM][BIN_NUM][BIN_NUM];
  #pragma omp parallel for schedule(dynamic, 1)
  for(size_t pic = 0; pic<pic_num; ++pic)
    for(size_t pi = 0; pi<keypoints[pic].size(); ++pi) {
      const Keypoint& p = keypoints[pic][pi];
      int idx[3];
      for(int i = 0; i<3; ++i) {
        double diff = (p.patch.at<double>(pts[i])-mean[i])/sd[i];
        if(diff <= -BOUND) idx[i] = -1;
        else if(diff >= BOUND) idx[i] = BIN_NUM-1;
        else idx[i] = int(diff+BOUND);
      }
      for(int i=0; i<2; ++i) for(int j=0; j<2; ++j) for(int k=0; k<2; ++k)
        if(in_mid(idx[0]+i) && in_mid(idx[1]+j) && in_mid(idx[2]+k))
          table[pic][idx[0]+i][idx[1]+j][idx[2]+k].push_back(pi);
    }

  vector<vector<list<tuple<int,int,double>>>> 
    pre_match(pic_num, vector<list<tuple<int,int,double>>>(pic_num));
  list<double> all_sec;
  #pragma omp parallel for
  for(size_t pic1 = 0; pic1<pic_num; ++pic1)
    for(size_t pic2 = pic1+1; pic2<pic_num; ++pic2) {
      for(int i=0; i<BIN_NUM; ++i)
        for(int j=0; j<BIN_NUM; ++j)
          for(int k=0; k<BIN_NUM; ++k)
            for(auto pi : table[pic1][i][j][k]) {
              const auto& ki = keypoints[pic1][pi];
              double fir = DBL_MAX, sec = DBL_MAX;
              int fir_j = -1;
              for(auto pj : table[pic2][i][j][k]) {
                const auto& kj = keypoints[pic2][pj];
                Mat diff = ki.patch - kj.patch;
                double err = sum(diff.mul(diff))[0];
                if(err < fir) sec = fir, fir_j = pj, fir = err;
                else if(err < sec) sec = err;
              }
              if(fir_j != -1 && sec != DBL_MAX && 
                 is_align(ki, keypoints[pic2][fir_j], _para[2])) {
                pre_match[pic1][pic2].emplace_back(pi, fir_j, fir);
                all_sec.push_back(sec);
              }
            }
      if(!panorama_mode) break;
    }
  double sec_mn = accumulate(all_sec.begin(), all_sec.end(), 0.0)/all_sec.size();

  #pragma omp parallel for
  for(size_t pic1 = 0; pic1<pic_num; ++pic1)
    for(size_t pic2 = pic1+1; pic2<pic_num; ++pic2) {
      pre_match[pic1][pic2].sort();
      pre_match[pic1][pic2].unique([](auto& x1, auto& x2) { 
                                return get<0>(x1) == get<0>(x2) && 
                                       get<1>(x1) == get<1>(x2); });
      auto it = pre_match[pic1][pic2].begin();
      for(; it != pre_match[pic1][pic2].end();)
        if(get<2>(*it) < THRESHOLD*sec_mn &&
           check_match_haar(*it, pic1, pic2, sec_mn)) ++it;
        else it = pre_match[pic1][pic2].erase(it);
      if(!panorama_mode) break;
    }

  match_pairs.clear();
  match_pairs.resize(pic_num, vector<vector<pair<int,int>>>(pic_num));
  #pragma omp parallel for
  for(size_t pic1 = 0; pic1<pic_num; ++pic1)
    for(size_t pic2 = pic1+1; pic2<pic_num; ++pic2) {
      match_pairs[pic1][pic2].resize(pre_match[pic1][pic2].size());
      int cnt = 0;
      for(const auto& it : pre_match[pic1][pic2])
        match_pairs[pic1][pic2][cnt++] = {get<0>(it), get<1>(it)};
      if(!panorama_mode) break;
    }
}
bool MATCHING::in_mid(const int& x) { return x >= 0 && x < int(_para[0])-1; };
bool MATCHING::is_align(const Keypoint& k1, const Keypoint& k2, 
                        const double& th) {
  return abs(k1.y - k2.y) < th;
}
void MATCHING::exhaustive() {
  namedWindow("exhaustive", WINDOW_NORMAL);
  cerr << __func__;
  size_t pic_num = keypoints.size();
  match_pairs.clear();
  match_pairs.resize(pic_num, vector<vector<pair<int,int>>>(pic_num));
  #pragma omp parallel for
  for(size_t pic1 = 0; pic1<pic_num; ++pic1)
    for(size_t pic2 = 0; pic2<pic_num; ++pic2) {
      for(size_t i = 0; i<keypoints[pic1].size(); ++i) {
        const auto& ki = keypoints[pic1][i];
        double min_err = DBL_MAX;
        int bestj = -1;
        for(size_t j = 0; j<keypoints[pic2].size(); ++j) {
          const auto& kj = keypoints[pic2][j];
          Mat diff = ki.patch - kj.patch;
          double err = sum(diff.mul(diff))[0];
          if(err < min_err) {
            min_err = err;
            bestj = j;
          }
        }
        if(bestj != -1 && check_match_exhaustive(i, bestj, pic1, pic2))
           //is_align(ki, keypoints[pic+1][bestj], _para[0]))
          match_pairs[pic1][pic2].emplace_back(i, bestj);
      }
      const auto red = Scalar(0, 0, 255);
      Mat img0 = imgs[pic1].clone();
      Mat img1 = imgs[pic2].clone();
      for (const auto& p : match_pairs[pic1][pic2]) {
        const Keypoint& kp0 = keypoints[pic1][p.first];
        const Keypoint& kp1 = keypoints[pic2][p.second];
        drawMarker(img0, Point(kp0.x, kp0.y), red, MARKER_CROSS, 20, 2);
        drawMarker(img1, Point(kp1.x, kp1.y), red, MARKER_CROSS, 20, 2);
      }
      Size sz[2] = {imgs[pic1].size(), imgs[pic2].size()};
      Mat show(sz[0].height, sz[0].width+sz[1].width, CV_8UC3);
      Mat left(show, Rect(0, 0, sz[0].width, sz[0].height));
      Mat right(show, Rect(sz[0].width, 0, sz[1].width, sz[1].height));
      img0.copyTo(left);
      img1.copyTo(right);
      for(const auto& p : match_pairs[pic1][pic2]) {
        const Keypoint& kp0 = keypoints[pic1][p.first];
        const Keypoint& kp1 = keypoints[pic2][p.second];
        line(show, Point(kp0.x, kp0.y), 
             Point(sz[0].width+kp1.x, kp1.y), red, 2, 8);
      }
      imshow("exhaustive", show);
      waitKey(0);
      if(!panorama_mode) break;
    }
}
bool MATCHING::check_match_haar(const tuple<int, int, double>& mp, 
                           size_t pic1, size_t pic2, double sec_mn) const {
  const int& pj = get<1>(mp);
  const auto& kj = keypoints[pic2][pj];
  double fir = DBL_MAX, sec = DBL_MAX;
  int fir_i = -1;
  #pragma omp parallel for
  for(size_t pi = 0; pi<keypoints[pic1].size(); ++pi) {
    const auto& ki = keypoints[pic1][pi];
    Mat diff = ki.patch - kj.patch;
    double err = sum(diff.mul(diff))[0];
    if(err < fir) sec = fir, fir_i = pi, fir = err;
    else if(err < sec) sec = err;
  }
  return fir_i == get<0>(mp) && fir < sec_mn * _para[1];
}
bool MATCHING::check_match_exhaustive(int target_i, int j, 
                                      size_t pic1, size_t pic2) const {
  const auto& kj = keypoints[pic2][j];
  double min_err = DBL_MAX;
  int best_i = -1;
  for(size_t i = 0; i<keypoints[pic1].size(); ++i) {
    const auto& ki = keypoints[pic1][i];
    Mat diff = kj.patch - ki.patch;
    double err = sum(diff.mul(diff))[0];
    if(err < min_err) {
      min_err = err;
      best_i = i;
    }
  }
  return best_i != -1 && target_i == best_i;
}
