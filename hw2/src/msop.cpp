#include <opencv2/opencv.hpp>
#include <iostream>
#include <climits>

using namespace cv;
using namespace std;

#include "msop.hpp"

#define MAX_LAYER           5
#define G_KERN              0
#define SIGMA_P             1.0   // pyramid smoothing
#define SIGMA_I             1.5   // integration scale
#define SIGMA_D             1.0   // derivative scale
#define HM_THRESHOLD        0.0005
#define ANMS_ROBUST_RATIO   0.9
#define KEYPOINT_NUM        500

struct PreKeypoint {
  int     x;
  int     y;
  int     minR2;
  float   hm;
  PreKeypoint() {}
  PreKeypoint(int x, int y, float h)
    : x(x), y(y), minR2(INT_MAX), hm(h) {}
};

bool is_greater_r(PreKeypoint i, PreKeypoint j) { 
  return (i.minR2 > j.minR2); 
}

void MSOP::process(const vector<Mat>& img_input) {
  namedWindow("process", WINDOW_NORMAL);
  vector<Mat> imgs(img_input);

  for (auto img : imgs) {
    vector<Mat> pyr;
    // image preprocessing
    cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32FC1);
    img *= 1./255;
    pyr.push_back(img);
    for (int i = 1; i < MAX_LAYER; ++i) {
      pyr.push_back(Mat());
      GaussianBlur(pyr[i-1], pyr[i], Size(G_KERN, G_KERN), SIGMA_P);
      resize(pyr[i], pyr[i], Size(), 0.5, 0.5);
    }

    // apply multi-scale Harris corner detector
    for (int i = 0; i < MAX_LAYER; ++i) {
      Mat P, Px, Py;
      Mat Kernel_x = (Mat_<float>(1, 3) << -0.5, 0, 0.5); 
      Mat Kernel_y = (Mat_<float>(3, 1) << -0.5, 0, 0.5); 
      GaussianBlur(pyr[i], P, Size(G_KERN, G_KERN), SIGMA_D);
      filter2D(P, Px, -1, Kernel_x);
      filter2D(P, Py, -1, Kernel_y);
      // H(x,y) = [Hxx Hxy; Hxy Hyy]
      Mat Hxx = Px.mul(Px);
      Mat Hxy = Px.mul(Py);
      Mat Hyy = Py.mul(Py);
      GaussianBlur(Hxx, Hxx, Size(G_KERN, G_KERN), SIGMA_I);
      GaussianBlur(Hxy, Hxy, Size(G_KERN, G_KERN), SIGMA_I);
      GaussianBlur(Hyy, Hyy, Size(G_KERN, G_KERN), SIGMA_I);
      Mat HM = (Hxx.mul(Hyy) - Hxy.mul(Hxy)) / (Hxx + Hyy);
      // compute keypoints
      Mat show = img;
      cvtColor(show, show, CV_GRAY2BGR);
      
      vector<PreKeypoint> kpts;
      for (int x = 1, xm = pyr[i].cols-1; x < xm; ++x)
        for (int y = 1, ym = pyr[i].rows-1; y < ym; ++y) {
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
          kpts.push_back(PreKeypoint(x, y, val));
        }
      // apply ANMS method
      for (int i = 0, n = kpts.size(); i < n; ++i)
        for (int j = i+1; j < n; ++j) {
          int newR2 = pow(kpts[i].x - kpts[j].x, 2) + pow(kpts[i].y - kpts[j].y, 2);
          if (kpts[i].hm < ANMS_ROBUST_RATIO * kpts[j].hm) {
            if (newR2 < kpts[i].minR2) kpts[i].minR2 = newR2; 
          } else if (kpts[j].hm < ANMS_ROBUST_RATIO * kpts[i].hm) {
            if (newR2 < kpts[j].minR2) kpts[j].minR2 = newR2;
          }
        } 
      sort(kpts.begin(), kpts.end(), is_greater_r);
      kpts.resize(KEYPOINT_NUM);
      for (auto p : kpts)
        drawMarker(show, Point(p.x, p.y), Scalar(0, 0, 255), 
          MARKER_CROSS, 20, 2);
      imshow("process", show);
      waitKey(0);      
    }
  }
}