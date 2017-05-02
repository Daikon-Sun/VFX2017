#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>

using namespace cv;
using namespace std;
using namespace std::chrono;

#include "msop.hpp"

#define MAX_LAYER           5
#define G_KERN              0
#define SIGMA_P             1.0   // pyramid smoothing
#define SIGMA_I             1.5   // integration scale
#define SIGMA_D             1.0   // derivative scale
#define SIGMA_O             4.5   // orientation scale
#define HM_THRESHOLD        0.0005
#define ANMS_ROBUST_RATIO   0.9
#define KEYPOINT_NUM        500
#define SAMPLE_SPACING      5

bool is_greater_r(PreKeypoint i, PreKeypoint j) { 
  return (i.minR2 > j.minR2); 
}
void execute(function<void ()> f) {
  using std::chrono::steady_clock;
  steady_clock::time_point start = steady_clock::now();
  f();
  steady_clock::time_point end = steady_clock::now();
  cerr << duration_cast<seconds>(end-start).count() << " seconds" << endl;
}

void MSOP::process(const vector<Mat>& img_input) {
  for(const auto& img : img_input) imgs.push_back(img.clone());
  pic_num = img_input.size();

  using namespace std::placeholders;
  execute(bind(&MSOP::detection, this));
  execute(bind(&MSOP::matching, this));
  execute(bind(&MSOP::warping, this));
  //execute(bind(&MSOP::visualize,this));
  execute(bind(&MSOP::RANSAC, this));
}
void MSOP::detection() {
  cerr << "start " << __func__ << "...";
  namedWindow("process", WINDOW_NORMAL);
  keypoints.clear();
  keypoints.resize(imgs.size(), vector<Keypoint>());

  #pragma omp parallel for
  for (size_t i = 0; i<imgs.size(); ++i) {
    Mat img = imgs[i].clone();
    vector<Mat> pyr;
    // image preprocessing
    cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32FC1);
    img *= 1./255;
    pyr.push_back(img.clone());
    pyr.resize(MAX_LAYER+1);
    for (int lev = 1; lev < MAX_LAYER+1; ++lev) {
      GaussianBlur(pyr[lev-1], pyr[lev], Size(G_KERN, G_KERN), SIGMA_P);
      resize(pyr[lev], pyr[lev], Size(), 0.5, 0.5);
    }

    // apply multi-scale Harris corner detector
    vector<Keypoint>& kpts = keypoints[i];
    kpts.reserve(MAX_LAYER * KEYPOINT_NUM);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int lev = 0; lev < MAX_LAYER; ++lev) {
      Mat P, Px, Py;
      GaussianBlur(pyr[lev], P, Size(G_KERN, G_KERN), SIGMA_D);
      filter2D(P, Px, -1, Kernel_x);
      filter2D(P, Py, -1, Kernel_y);
      Mat Hxx = Px.mul(Px);
      Mat Hxy = Px.mul(Py);
      Mat Hyy = Py.mul(Py);
      GaussianBlur(Hxx, Hxx, Size(G_KERN, G_KERN), SIGMA_I);
      GaussianBlur(Hxy, Hxy, Size(G_KERN, G_KERN), SIGMA_I);
      GaussianBlur(Hyy, Hyy, Size(G_KERN, G_KERN), SIGMA_I);
      Mat HM = (Hxx.mul(Hyy) - Hxy.mul(Hxy)) / (Hxx + Hyy);

      // compute keypoints
      //Mat show = img;
      //cvtColor(show, show, CV_GRAY2BGR);
      vector<PreKeypoint> pre_kpts;
      for (int x = 60, xm = pyr[lev].cols-60; x < xm; ++x)
        for (int y = 60, ym = pyr[lev].rows-60; y < ym; ++y) {
          float val = HM.at<float>(y,x);
          if (val < HM_THRESHOLD) continue;
          if ( 
            val < HM.at<float>(y+1,x+1) ||
            val < HM.at<float>(y+1,x  ) ||
            val < HM.at<float>(y+1,x-1) ||
            val < HM.at<float>(y  ,x+1) ||
            val < HM.at<float>(y  ,x-1) ||
            val < HM.at<float>(y-1,x+1) ||
            val < HM.at<float>(y-1,x  ) ||
            val < HM.at<float>(y-1,x-1)
          ) continue;
          pre_kpts.push_back(PreKeypoint(x, y, val));
        }

      // apply ANMS method
      for (int k = 0, n = pre_kpts.size(); k < n; ++k)
        for (int j = i+1; j < n; ++j) {
          int newR2 = pow(pre_kpts[k].x - pre_kpts[j].x, 2) 
                    + pow(pre_kpts[k].y - pre_kpts[j].y, 2);
          if (pre_kpts[k].hm < ANMS_ROBUST_RATIO * pre_kpts[j].hm) {
            if (newR2 < pre_kpts[k].minR2) pre_kpts[k].minR2 = newR2; 
          } else if (pre_kpts[j].hm < ANMS_ROBUST_RATIO * pre_kpts[k].hm) {
            if (newR2 < pre_kpts[j].minR2) pre_kpts[j].minR2 = newR2;
          }
        } 
      sort(pre_kpts.begin(), pre_kpts.end(), is_greater_r);
      if (pre_kpts.size() > KEYPOINT_NUM)
        pre_kpts.resize(KEYPOINT_NUM);

      // sub-pixel accuracy and orientation assignment
      GaussianBlur(pyr[lev], P, Size(G_KERN, G_KERN), SIGMA_O);
      filter2D(P, Px, -1, Kernel_x);
      filter2D(P, Py, -1, Kernel_y);
      size_t j = kpts.size();
      for (const auto& p : pre_kpts) {
        float dx = (
          HM.at<float>(p.y, p.x+1) -
          HM.at<float>(p.y, p.x-1)
        ) / 2;
        float dy = (
          HM.at<float>(p.y+1, p.x) -
          HM.at<float>(p.y-1, p.x)
        ) / 2;
        float dxx = (
          HM.at<float>(p.y, p.x+1) + 
          HM.at<float>(p.y, p.x-1) -
          HM.at<float>(p.y, p.x  ) * 2
        );
        float dyy = (
          HM.at<float>(p.y+1, p.x) + 
          HM.at<float>(p.y-1, p.x) -
          HM.at<float>(p.y  , p.x) * 2
        );
        float dxy = (
          HM.at<float>(p.y+1, p.x+1) + 
          HM.at<float>(p.y-1, p.x-1) -
          HM.at<float>(p.y+1, p.x-1) -
          HM.at<float>(p.y-1, p.x+1)
        ) / 4;
        Mat m1 = (Mat_<float>(2,2) << dxx, dxy, dxy, dyy);
        Mat m2 = (Mat_<float>(2,1) << dx, dy);
        Mat xm = m1.inv() * m2;
        float x = p.x - xm.at<float>(0,0);
        float y = p.y - xm.at<float>(0,1);
        float ux = 
          Px.at<float>(ceil(y) , ceil(x) ) * (floor(x) - x) * (floor(y) - y) +
          Px.at<float>(ceil(y) , floor(x)) * (x - ceil(x) ) * (floor(y) - y) +
          Px.at<float>(floor(y), ceil(x) ) * (floor(x) - x) * (y - ceil(y) ) +
          Px.at<float>(floor(y), floor(x)) * (x - ceil(x) ) * (y - ceil(y) );
        float uy = 
          Py.at<float>(ceil(y) , ceil(x) ) * (floor(x) - x) * (floor(y) - y) +
          Py.at<float>(ceil(y) , floor(x)) * (x - ceil(x) ) * (floor(y) - y) +
          Py.at<float>(floor(y), ceil(x) ) * (floor(x) - x) * (y - ceil(y) ) +
          Py.at<float>(floor(y), floor(x)) * (x - ceil(x) ) * (y - ceil(y) );
        kpts.emplace_back(x, y, lev, atan2(uy,ux)*180/M_PI);
      }

      // compute feature descriptor
      for(; j<kpts.size(); ++j) {
        auto& p = kpts[j];
        Mat m, rot;
        GaussianBlur(pyr[lev+1], m, Size(G_KERN, G_KERN), SIGMA_P);
        rot = getRotationMatrix2D(Point(p.x/2, p.y/2), p.t, 1);
        warpAffine(m, m, rot, m.size());
        getRectSubPix(m, Size(40, 40), Point(p.x/2, p.y/2), p.patch);
        resize(p.patch, p.patch, Size(), 0.2, 0.2);
        Scalar mean, sd;
        meanStdDev(p.patch, mean, sd);
        p.patch = (p.patch-mean[0])/sd[0];
        p.patch = HAAR * p.patch * HAAR_T;
        p.x *= pow(2, p.l);
        p.y *= pow(2, p.l);
      }
    }
    tot_kpts += kpts.size();
  }
}
bool MSOP::is_align(const Keypoint& k1, const Keypoint& k2) {
  return abs(k1.y - k2.y) < Y_MAX_DIFF;
}
bool MSOP::check_match(const tuple<int, int, float>& mp, 
                       size_t pic, float sec_mn) const {
  const int& pj = get<1>(mp);
  const auto& kj = keypoints[pic+1][pj];
  float fir = FLT_MAX, sec = FLT_MAX;
  int fir_i = -1;
  #pragma omp parallel for
  for(size_t pi = 0; pi<keypoints[pic].size(); ++pi) {
    const auto& ki = keypoints[pic][pi];
    Mat diff = ki.patch - kj.patch;
    float err = sum(diff.mul(diff))[0];
    if(err < fir) sec = fir, fir_i = pi, fir = err;
    else if(err < sec) sec = err;
  }
  return fir_i == get<0>(mp) && fir < sec_mn * THRESHOLD;
}
void MSOP::matching() {
  cerr << "start " << __func__ << "...";
  Point pts[3] = {Point(1, 0), Point(0, 1), Point(1, 1)};

  //find mean and std for the first three nonzero Haar wavelet coefficient
  float mean[3], sd[3];
  #pragma omp parallel for
  for(int i = 0; i<3; ++i) {
    vector<float> v(tot_kpts), diff(tot_kpts);
    size_t cnt = 0;
    for(size_t pic = 0; pic<pic_num; ++pic)
      for(const auto& p:keypoints[pic])
        v[cnt++] = p.patch.at<float>(pts[i]);
    mean[i] = std::accumulate(v.begin(), v.end(), 0.0) / tot_kpts;
   	transform(v.begin(), v.end(), diff.begin(), 
		          [mean, i](const float& v) { return v-mean[i]; });
    sd[i] = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    sd[i] = sqrtf(sd[i] / diff.size()) * 6.0 / BIN_NUM;
  }

  //put keypoints into bins
  list<int> table[pic_num][BIN_NUM][BIN_NUM][BIN_NUM];
  #pragma omp parallel for schedule(dynamic, 1)
  for(size_t pic = 0; pic<pic_num; ++pic)
    for(size_t pi = 0; pi<keypoints[pic].size(); ++pi) {
      const Keypoint& p = keypoints[pic][pi];
      int idx[3];
      for(int i = 0; i<3; ++i) {
        float diff = (p.patch.at<float>(pts[i])-mean[i])/sd[i];
        if(diff <= -BOUND) idx[i] = -1;
        else if(diff >= BOUND) idx[i] = BIN_NUM-1;
        else idx[i] = int(diff+BOUND);
      }
      for(int i=0; i<2; ++i) for(int j=0; j<2; ++j) for(int k=0; k<2; ++k)
        if(in_mid(idx[0]+i) && in_mid(idx[1]+j) && in_mid(idx[2]+k))
          table[pic][idx[0]+i][idx[1]+j][idx[2]+k].push_back(pi);
    }
  
  //match keypoints
  vector< list< tuple<int, int, float> > > pre_match(pic_num-1);
  list<float> all_sec;
  #pragma omp parallel for collapse(4)
  for(size_t pic = 0; pic<pic_num-1; ++pic)
    for(int i=0; i<BIN_NUM; ++i)
      for(int j=0; j<BIN_NUM; ++j)
        for(int k=0; k<BIN_NUM; ++k)
          for(auto pi : table[pic][i][j][k]) {
            const auto& ki = keypoints[pic][pi];
            float fir = FLT_MAX, sec = FLT_MAX;
            int fir_j = -1;
            for(auto pj : table[pic+1][i][j][k]) {
              const auto& kj = keypoints[pic+1][pj];
              Mat diff = ki.patch - kj.patch;
              float err = sum(diff.mul(diff))[0];
              if(err < fir) sec = fir, fir_j = pj, fir = err;
              else if(err < sec) sec = err;
            }
            if(fir_j != -1 && sec != FLT_MAX && 
               is_align(ki, keypoints[pic+1][fir_j])) {
              pre_match[pic].emplace_back(pi, fir_j, fir);
              all_sec.push_back(sec);
            }
          }
  float sec_mn = accumulate(all_sec.begin(), all_sec.end(), 0.0)/all_sec.size();

  //remove duplicate
  //Feature-Space Outlier Rejection based on averaged 2-NN and two-way check
  #pragma omp parallel for schedule(dynamic, 1)
  for(size_t pic = 0; pic<pic_num-1; ++pic) {
    pre_match[pic].sort();
    pre_match[pic].unique([](auto& x1, auto& x2) { 
                              return get<0>(x1) == get<0>(x2) && 
                                     get<1>(x1) == get<1>(x2); });
    for(auto it = pre_match[pic].begin(); it != pre_match[pic].end();)
      if(get<2>(*it) < THRESHOLD*sec_mn && check_match(*it, pic, sec_mn)) ++it;
      else it = pre_match[pic].erase(it);
  }
  match_pairs.clear();
  match_pairs.resize(pic_num-1);
  for(size_t pic = 0; pic<pic_num-1; ++pic) {
    match_pairs[pic].resize(pre_match[pic].size());
    int cnt = 0;
    for(const auto& it : pre_match[pic])
      match_pairs[pic][cnt++] = {get<0>(it), get<1>(it)};
  }
}
void MSOP::visualize() {
  cerr << "start " << __func__ << "...";
  //visualize feature matching
  for(size_t pic = 0; pic+1<keypoints.size(); ++pic) {
    const auto red = Scalar(0, 0, 255);
    Mat img0 = imgs[pic].clone();
    Mat img1 = imgs[pic+1].clone();
    for (const auto& p : match_pairs[pic]) {
      const Keypoint& kp0 = keypoints[pic][p.first];
      const Keypoint& kp1 = keypoints[pic+1][p.second];
      drawMarker(img0, Point(kp0.x, kp0.y), red, MARKER_CROSS, 20, 2);
      drawMarker(img1, Point(kp1.x, kp1.y), red, MARKER_CROSS, 20, 2);
    }
    Size sz[2];
    for(size_t i = 0; i<2; ++i) sz[i] = imgs[pic+i].size();
    Mat show(sz[0].height, sz[0].width+sz[1].width, CV_8UC3);
    Mat left(show, Rect(0, 0, sz[0].width, sz[0].height));
    Mat right(show, Rect(sz[0].width, 0, sz[1].width, sz[1].height));
    img0.copyTo(left);
    img1.copyTo(right);
    for(const auto& p : match_pairs[pic]) {
      const Keypoint& kp0 = keypoints[pic][p.first];
      const Keypoint& kp1 = keypoints[pic+1][p.second];
      line(show, Point(kp0.x, kp0.y), 
           Point(sz[0].width+kp1.x, kp1.y), red, 2, 8);
    }
    imshow("process", show);
    waitKey(0);
  }
}
pair<float, float> MSOP::projected_xy(float w, float h, float x, float y) {
  return {F*atanf((x-w/2)/F) + w/2, F*(y-h/2)/sqrtf(F*F+(x-w/2)*(x-w/2)) + h/2};
}
void MSOP::warping() {
  cerr << "start " << __func__ << "...";
  Size sz = imgs[0].size();
  //image warping
  for(size_t i = 0; i<imgs.size(); ++i) {
    Mat img = Mat(sz, CV_8UC3, Scalar(0, 0, 0));
    for(int y = 0; y<sz.height; ++y) for(int x = 0; x<sz.width; ++x) {
      int nx, ny; tie(nx, ny) = projected_xy(sz.width, sz.height, x, y);
      img.at<Vec3b>(ny, nx) = imgs[i].at<Vec3b>(y, x); 
    }
    img.copyTo(imgs[i]);
  }
  //feature point warping
  for(size_t i = 0; i<keypoints.size(); ++i)
    for(size_t j = 0; j<keypoints[i].size(); ++j)
      tie(keypoints[i][j].x, keypoints[i][j].y) = 
        projected_xy(sz.width, sz.height, keypoints[i][j].x, keypoints[i][j].y);
}
bool MSOP::is_inliner(size_t pic, float sx, float sy, pair<int, int>& kpid) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[pic][kpid1], kp2 = keypoints[pic+1][kpid2];
  float _sx = kp1.x - kp2.x, _sy = kp1.y - kp2.y;
  return (sx-_sx) * (sx-_sx) + (sy-_sy) * (sy-_sy) < RANSAC_THRESHOLD;
}
void MSOP::RANSAC() { 
  cerr << "start " << __func__ << "...";
  shift.clear();
  shift.resize(pic_num-1);
  for(size_t pic = 0; pic<match_pairs.size(); ++pic) {
    int best_cnt = 0, best_pair, best_sx, best_sy;
    for(int i = 0; i<RANSAC_K; ++i) {
      size_t id1 = rand()%match_pairs[pic].size();
      int kpid1, kpid2; tie(kpid1, kpid2) = match_pairs[pic][id1];
      const auto& kp1 = keypoints[pic][kpid1], kp2 = keypoints[pic+1][kpid2];
      float sx = kp1.x - kp2.x, sy = kp1.y - kp2.y;
      int in_cnt = 0;
      for(size_t id2 = 0; id2<match_pairs[pic].size(); ++id2) if(id1 != id2)
        in_cnt += is_inliner(pic, sx, sy, match_pairs[pic][id2]);
      if(in_cnt > best_cnt) {
        best_cnt = in_cnt;
        best_pair = id1;
        best_sx = sx;
        best_sy = sy;
      }
    }
    shift[pic] = {best_sx, best_sy};
    cerr << pic << " " << best_cnt << " " << best_sx << " " << best_sy << endl;
  }
}
