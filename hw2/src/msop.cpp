#include <opencv2/opencv.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;

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

void MSOP::process(const vector<Mat>& img_input) {
  namedWindow("process", WINDOW_NORMAL);
  keypoints.resize(img_input.size(), vector<Keypoint>());

  #pragma omp parallel for
  for (size_t i = 0; i<img_input.size(); ++i) {
    Mat img = img_input[i].clone();
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
      //for (const auto& p : pre_kpts)
      //  drawMarker(show, Point(p.x, p.y), Scalar(0, 0, 255), 
      //    MARKER_CROSS, 20, 2);
      //imshow("process", show);
      //waitKey(0);

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
      }
    }
    tot_kpts += kpts.size();
  }
  cerr << "start matching..." << endl;
  matching(img_input);
}
bool MSOP::is_align(const Keypoint& k1, const Keypoint& k2) {
  return abs(k1.t_y() - k2.t_y()) < Y_MAX_DIFF;
}
void MSOP::matching(const vector<Mat>& img_input) {
  size_t pic_num = img_input.size();
  Point pts[3] = {Point(1, 0), Point(0, 1), Point(1, 1)};

  //find mean and std for the first three nonzero Haar wavelet coefficient
  float mean[3], sd[3];
  #pragma omp parallel for schedule(dynamic, 1)
  for(int i = 0; i<3; ++i) {
    vector<float> v(tot_kpts), diff(tot_kpts);
    size_t cnt = 0;
    for(size_t pic = 0; pic<pic_num; ++pic)
      for(const auto& p:keypoints[pic])
        v[cnt++] = p.patch.at<float>(pts[i]);
    mean[i] = std::accumulate(v.begin(), v.end(), 0.0) / tot_kpts;
   	transform(v.begin(), v.end(), diff.begin(), 
		          [mean, i](const float& x) { return x-mean[i]; });
    sd[i] = sqrt(inner_product(diff.begin(), diff.end(), diff.begin(), 0.0));
    sd[i] /= (BIN_NUM-1)/6.;
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
  list< tuple<int, int, float> > match_pairs[pic_num-1];
  list<float> all_sec;
  #pragma omp parallel for collapse(4)
  for(size_t pic = 0; pic<pic_num-1; ++pic) {
    for(int i=0; i<BIN_NUM; ++i)
      for(int j=0; j<BIN_NUM; ++j)
        for(int k=0; k<BIN_NUM; ++k) {
          for(auto pi : table[pic][i][j][k]) {
            const auto& ki = keypoints[pic][pi];
            float fir = FLT_MAX, sec = FLT_MAX;
            int fir_i = -1;
            for(auto pj : table[pic+1][i][j][k]) {
              const auto& kj = keypoints[pic+1][pj];
              Mat diff = ki.patch - kj.patch;
              float err = sum(diff.mul(diff))[0];
              if(err < fir) sec = fir, fir_i = pj, fir = err;
              else if(err < sec) sec = err;
            }
            if(fir_i != -1 && sec != FLT_MAX && 
               is_align(ki, keypoints[pic+1][fir_i])) {
              match_pairs[pic].emplace_back(pi, fir_i, fir);
              all_sec.push_back(sec);
            }
          }
        }
  }
  float sec_mn = accumulate(all_sec.begin(), all_sec.end(), 0.0)/all_sec.size();

  //Feature-Space Outlier Rejection based on averaged 2-NN
  #pragma omp parallel for schedule(dynamic, 1)
  for(size_t pic = 0; pic<pic_num-1; ++pic)
    for(auto it = match_pairs[pic].begin(); it!=match_pairs[pic].end(); )
      if(get<2>(*it) < THRESHOLD*sec_mn) ++it;
      else it = match_pairs[pic].erase(it);

  //visualize feature matching
  for(size_t pic = 0; pic<pic_num-1; ++pic) {
    const auto red = Scalar(0, 0, 255);
    Mat img0 = img_input[pic].clone();
    Mat img1 = img_input[pic+1].clone();
    for (const auto& p : match_pairs[pic]) {
      const Keypoint& kp0 = keypoints[pic][get<0>(p)];
      const Keypoint& kp1 = keypoints[pic+1][get<1>(p)];
      drawMarker(img0, Point(kp0.t_x(), kp0.t_y()), red, MARKER_CROSS, 20, 2);
      drawMarker(img1, Point(kp1.t_x(), kp1.t_y()), red, MARKER_CROSS, 20, 2);
    }
    Size sz[2];
    for(size_t i = 0; i<2; ++i) sz[i] = img_input[pic+i].size();
    Mat show(sz[0].height, sz[0].width+sz[1].width, CV_8UC3);
    Mat left(show, Rect(0, 0, sz[0].width, sz[0].height));
    Mat right(show, Rect(sz[0].width, 0, sz[1].width, sz[1].height));
    img0.copyTo(left);
    img1.copyTo(right);
    for(const auto& p : match_pairs[pic]) {
      const Keypoint& kp0 = keypoints[pic][get<0>(p)];
      const Keypoint& kp1 = keypoints[pic+1][get<1>(p)];
      line(show, Point(kp0.t_x(), kp0.t_y()), 
           Point(sz[0].width+kp1.t_x(), kp1.t_y()), red, 2, 8);
    }
    imshow("process", show);
    waitKey(0);
  }
}
