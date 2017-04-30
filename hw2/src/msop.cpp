#include <opencv2/opencv.hpp>
#include <iostream>
#include <climits>
#include <cmath>

using namespace cv;
using namespace std;

#include "msop.hpp"
#include "type.hpp"

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
  vector<Mat> imgs(img_input);

  for (auto& img : imgs) {
    vector<Mat> pyr;
    // image preprocessing
    cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32FC1);
    img *= 1./255;
    pyr.push_back(img);
    for (int lev = 1; lev < MAX_LAYER+1; ++lev) {
      pyr.push_back(Mat());
      GaussianBlur(pyr[lev-1], pyr[lev], Size(G_KERN, G_KERN), SIGMA_P);
      resize(pyr[lev], pyr[lev], Size(), 0.5, 0.5);
    }

    // apply multi-scale Harris corner detector
    for (int lev = 0; lev < MAX_LAYER; ++lev) {
      Mat P, Px, Py;
      Mat Kernel_x = (Mat_<float>(1, 3) << -0.5, 0, 0.5); 
      Mat Kernel_y = (Mat_<float>(3, 1) << -0.5, 0, 0.5); 
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
      Mat show = img;
      cvtColor(show, show, CV_GRAY2BGR);
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
      for (int i = 0, n = pre_kpts.size(); i < n; ++i)
        for (int j = i+1; j < n; ++j) {
          int newR2 = pow(pre_kpts[i].x - pre_kpts[j].x, 2) + pow(pre_kpts[i].y - pre_kpts[j].y, 2);
          if (pre_kpts[i].hm < ANMS_ROBUST_RATIO * pre_kpts[j].hm) {
            if (newR2 < pre_kpts[i].minR2) pre_kpts[i].minR2 = newR2; 
          } else if (pre_kpts[j].hm < ANMS_ROBUST_RATIO * pre_kpts[i].hm) {
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
      vector<Keypoint> kpts;
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
        kpts.push_back(Keypoint(x, y, lev, atan2(uy,ux)*180/M_PI));
      }

      // compute feature descriptor
      cerr << "num of kpts " << kpts.size() << endl;
      for(auto& p : kpts) {
        Mat m, rot;
        GaussianBlur(pyr[lev+1], m, Size(G_KERN, G_KERN), SIGMA_P);
        //cerr << p.x << " " << p.y << " " << p.t << endl;
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
  }
}
