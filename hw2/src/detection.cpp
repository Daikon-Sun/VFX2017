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

#include "detection.hpp"

constexpr int MAX_LAYER           = 1;
constexpr float G_KERN            = 0;
constexpr float SIGMA_P           = 1.0;  // pyramid smoothing
constexpr float SIGMA_I           = 1.5;  // integration scale
constexpr float SIGMA_D           = 1.0;  // derivative scale
constexpr float SIGMA_O           = 4.5;  // orientation scale
constexpr float HM_THRESHOLD      = 0.0005;
constexpr float ANMS_ROBUST_RATIO = 0.9;
constexpr int KEYPOINT_NUM        = 500;
constexpr float SAMPLE_SPACING    = 5;

bool is_greater_r(PreKeypoint i, PreKeypoint j) { 
  return (i.minR2 > j.minR2); 
}
void DETECTION::MSOP() {
  cerr << __func__;
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
        p.x *= pow(2, p.l);
        p.y *= pow(2, p.l);
      }
    }
  }
}
//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//using namespace std;
//
//#include "sift.hpp"
//
//#define OCTAVE_LAYER            5       
//#define OCTAVE_NUM              3
//#define OCTAVE_SCALE            3
//#define GAUSSIAN_KERN           7
//#define PRE_SIGMA               1.6
//#define SIGMA                   pow(2, 1.0/(double)OCTAVE_SCALE)
//#define CONTRAST_THRESHOLD      0.3
//#define CURVATURE_THRESHOLD_R   10.0
//#define CURVATURE_THRESHOLD     pow(CURVATURE_THRESHOLD_R+1, 2)/CURVATURE_THRESHOLD_R
//
//inline bool is_extrema(const vector<vector<Mat>>&, int, int, int, int);
//
//vector<Mat> SIFT::process(const vector<Mat>& img) {
//  vector<Mat> L(img);
//
//  #ifdef SHOW_PROCESS
//  namedWindow("process", WINDOW_AUTOSIZE);
//  #endif
//
//  for (int i=0, n=img.size(); i<n; ++i) {
//    // preprocessing images
//    cvtColor(L[i], L[i], CV_BGR2GRAY);
//    L[i].convertTo(L[i], CV_64FC1);
//    L[i] *= 1./255;
//
//    /**************************************/
//    /** detection of scale-space extrema **/
//    /**************************************/
//    vector<vector<Mat>> g_octaves(OCTAVE_NUM);  // Gaussian octaves
//    vector<vector<Mat>> d_octaves(OCTAVE_NUM);  // DoG octaves
//    for (int t=0; t<OCTAVE_NUM; ++t) {
//      g_octaves[t].resize(OCTAVE_LAYER);
//      d_octaves[t].resize(OCTAVE_LAYER-1);
//      resize(L[i], g_octaves[t][0], Size(), 2*pow(0.5,t), 2*pow(0.5,t));
//      // compute Gaussian octaves
//      for (int l=0; l<OCTAVE_LAYER; ++l) {
//        double sigma = PRE_SIGMA * pow(SIGMA, l) * pow(2, t);
//        GaussianBlur(
//          g_octaves[t][0],
//          g_octaves[t][l],
//          Size((int)(sigma*4)*2+1, (int)(sigma*4)*2+1),
//          sigma
//        );
//      }
//      // comput DoG octaves
//      for (int l=0; l<OCTAVE_LAYER-1; ++l)
//        d_octaves[t][l] = g_octaves[t][l+1] - g_octaves[t][l];
//    }
//
//    /**************************************/
//    /**  accurate keypoint localization  **/
//    /**************************************/
//    int lim = (int)(1+(1+2*GAUSSIAN_KERN*SIGMA)/5);
//    for (int t=0; t<OCTAVE_NUM; ++t)
//      for (int l=1, l_max=OCTAVE_LAYER-2; l<l_max; ++l) {
//
//        #ifdef SHOW_PROCESS
//        Mat marked_img = g_octaves[t][l];
//        marked_img.convertTo(marked_img, CV_32FC1);
//        cvtColor(marked_img, marked_img, CV_GRAY2BGR);
//        #endif
//        
//        for (int c=lim, c_max=d_octaves[t][l].cols-lim; c<c_max; ++c)
//          for (int r=lim, r_max=d_octaves[t][l].rows-lim; r<r_max; ++r) {
//            if (!is_extrema(d_octaves, t, l, r, c)) continue;
//            double value = d_octaves[t][l].at<double>(r, c);
//            cerr << value << endl;
//            // thow out low contrast
//            double Dx = (
//              d_octaves[t][l].at<double>(r, c+1) - 
//              d_octaves[t][l].at<double>(r, c-1)
//            ) / 2;
//            double Dy = (
//              d_octaves[t][l].at<double>(r+1, c) - 
//              d_octaves[t][l].at<double>(r-1, c)
//            ) / 2;
//            double Ds = (
//              d_octaves[t][l+1].at<double>(r, c) - 
//              d_octaves[t][l-1].at<double>(r, c)
//            ) / 2;
//            double Dxx = (
//              d_octaves[t][l].at<double>(r, c-1) + 
//              d_octaves[t][l].at<double>(r, c+1) -
//              d_octaves[t][l].at<double>(r, c  ) * 2
//            );
//            double Dyy = (
//              d_octaves[t][l].at<double>(r-1, c) + 
//              d_octaves[t][l].at<double>(r+1, c) -
//              d_octaves[t][l].at<double>(r  , c) * 2
//            );
//            double Dss = (
//              d_octaves[t][l-1].at<double>(r, c) + 
//              d_octaves[t][l+1].at<double>(r, c) -
//              d_octaves[t][l  ].at<double>(r, c) * 2
//            );
//            double Dxy = (
//              d_octaves[t][l].at<double>(r-1, c-1) +
//              d_octaves[t][l].at<double>(r+1, c+1) -
//              d_octaves[t][l].at<double>(r-1, c+1) -
//              d_octaves[t][l].at<double>(r+1, c-1)
//            ) / 4;
//            double Dxs = (
//              d_octaves[t][l-1].at<double>(r, c-1) +
//              d_octaves[t][l+1].at<double>(r, c+1) -
//              d_octaves[t][l+1].at<double>(r, c-1) -
//              d_octaves[t][l-1].at<double>(r, c+1)
//            ) / 4;
//            double Dys = (
//              d_octaves[t][l-1].at<double>(r-1, c) +
//              d_octaves[t][l+1].at<double>(r+1, c) -
//              d_octaves[t][l+1].at<double>(r-1, c) -
//              d_octaves[t][l-1].at<double>(r+1, c)
//            ) / 4;
//            Mat H = (Mat_<double>(3,3) << 
//              Dxx, Dxy, Dxs, 
//              Dxy, Dyy, Dys,
//              Dxs, Dys, Dss
//            );
//            Mat parD = (Mat_<double>(3,1) << Dx, Dy, Ds);
//            Mat x = (Mat_<double>(3,1) << c, r, SIGMA);
//            Mat h = (-1) * H.inv() * parD;
//            value = value + 0.5 * parD.dot(h);
//            // if (value < 7) continue;
//            cerr << value << endl;
//            // eliminate edge responses; H: Hessian
//            double TrH = Dxx + Dyy;
//            double DetH = Dxx * Dyy - pow(Dxy, 2);
//            /*if (DetH == 0) continue;
//            if (TrH*TrH/DetH > CURVATURE_THRESHOLD) {
//              cerr << "eliminate edge" << endl;
//              continue;
//            }*/
//
//            #ifdef SHOW_PROCESS
//            drawMarker(marked_img, Point(c, r), Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
//            #endif
//          }
//        #ifdef SHOW_PROCESS
//        imshow("process", marked_img);
//        #endif
//        waitKey(0);
//      }
//  }
//
//    /**************************************/
//    /**      orientation assignment      **/
//    /**************************************/
//
//
//    /**************************************/
//    /**       keypoint descriptor        **/
//    /**************************************/
//
//}
//
//// helper functions
//inline bool is_extrema(const vector<vector<Mat>>& img, int t, int l, int r, int c) {
//  double value = img[t][l].at<double>(r, c);
//  /*Mat low = (Mat_<double>(3, 3) <<
//    img[t][l-1].at<double>(r+1, c-1),
//    img[t][l-1].at<double>(r+1, c  ),
//    img[t][l-1].at<double>(r+1, c+1),
//    img[t][l-1].at<double>(r  , c-1),
//    img[t][l-1].at<double>(r  , c  ),
//    img[t][l-1].at<double>(r  , c+1),
//    img[t][l-1].at<double>(r-1, c-1),
//    img[t][l-1].at<double>(r-1, c  ),
//    img[t][l-1].at<double>(r-1, c+1)
//  );
//
//  Mat mid = (Mat_<double>(3, 3) <<
//    img[t][l].at<double>(r+1, c-1),
//    img[t][l].at<double>(r+1, c  ),
//    img[t][l].at<double>(r+1, c+1),
//    img[t][l].at<double>(r  , c-1),
//    img[t][l].at<double>(r  , c  ),
//    img[t][l].at<double>(r  , c+1),
//    img[t][l].at<double>(r-1, c-1),
//    img[t][l].at<double>(r-1, c  ),
//    img[t][l].at<double>(r-1, c+1)
//  );
//
//  Mat high = (Mat_<double>(3, 3) <<
//    img[t][l+1].at<double>(r+1, c-1),
//    img[t][l+1].at<double>(r+1, c  ),
//    img[t][l+1].at<double>(r+1, c+1),
//    img[t][l+1].at<double>(r  , c-1),
//    img[t][l+1].at<double>(r  , c  ),
//    img[t][l+1].at<double>(r  , c+1),
//    img[t][l+1].at<double>(r-1, c-1),
//    img[t][l+1].at<double>(r-1, c  ),
//    img[t][l+1].at<double>(r-1, c+1)
//  );
//
//  Mat tmp = (Mat_<double>(27, 1) <<
//    img[t][l].at<double>(r+1, c+1),
//    img[t][l].at<double>(r+1, c  ),
//    img[t][l].at<double>(r+1, c-1),
//    img[t][l].at<double>(r  , c+1),
//    img[t][l].at<double>(r  , c  ),
//    img[t][l].at<double>(r  , c-1),
//    img[t][l].at<double>(r-1, c+1),
//    img[t][l].at<double>(r-1, c  ),
//    img[t][l].at<double>(r-1, c-1),
//    img[t][l-1].at<double>(r+1, c+1),
//    img[t][l-1].at<double>(r+1, c  ),
//    img[t][l-1].at<double>(r+1, c-1),
//    img[t][l-1].at<double>(r  , c+1),
//    img[t][l-1].at<double>(r  , c  ),
//    img[t][l-1].at<double>(r  , c-1),
//    img[t][l-1].at<double>(r-1, c+1),
//    img[t][l-1].at<double>(r-1, c  ),
//    img[t][l-1].at<double>(r-1, c-1),
//    img[t][l+1].at<double>(r+1, c+1),
//    img[t][l+1].at<double>(r+1, c  ),
//    img[t][l+1].at<double>(r+1, c-1),
//    img[t][l+1].at<double>(r  , c+1),
//    img[t][l+1].at<double>(r  , c  ),
//    img[t][l+1].at<double>(r  , c-1),
//    img[t][l+1].at<double>(r-1, c+1),
//    img[t][l+1].at<double>(r-1, c  ),
//    img[t][l+1].at<double>(r-1, c-1)
//  );
//  double min, max, minLoc, maxLoc;
//  minMaxLoc(tmp, &min, &max);
//  cerr << high << endl << mid << endl << low << endl;
//  cerr << value 
//       << " min:" << min
//       << " max:" << max << endl;
//  waitKey(0);*/
//  
//  return ((
//    value <= img[t][l].at<double>(r+1, c+1) &&
//    value <= img[t][l].at<double>(r+1, c  ) &&
//    value <= img[t][l].at<double>(r+1, c-1) &&
//    value <= img[t][l].at<double>(r  , c+1) &&
//    value <= img[t][l].at<double>(r  , c-1) &&
//    value <= img[t][l].at<double>(r-1, c+1) &&
//    value <= img[t][l].at<double>(r-1, c  ) &&
//    value <= img[t][l].at<double>(r-1, c-1) &&
//    value <= img[t][l-1].at<double>(r+1, c+1) &&
//    value <= img[t][l-1].at<double>(r+1, c  ) &&
//    value <= img[t][l-1].at<double>(r+1, c-1) &&
//    value <= img[t][l-1].at<double>(r  , c+1) &&
//    value <= img[t][l-1].at<double>(r  , c  ) &&
//    value <= img[t][l-1].at<double>(r  , c-1) &&
//    value <= img[t][l-1].at<double>(r-1, c+1) &&
//    value <= img[t][l-1].at<double>(r-1, c  ) &&
//    value <= img[t][l-1].at<double>(r-1, c-1) &&
//    value <= img[t][l+1].at<double>(r+1, c+1) &&
//    value <= img[t][l+1].at<double>(r+1, c  ) &&
//    value <= img[t][l+1].at<double>(r+1, c-1) &&
//    value <= img[t][l+1].at<double>(r  , c+1) &&
//    value <= img[t][l+1].at<double>(r  , c  ) &&
//    value <= img[t][l+1].at<double>(r  , c-1) &&
//    value <= img[t][l+1].at<double>(r-1, c+1) &&
//    value <= img[t][l+1].at<double>(r-1, c  ) &&
//    value <= img[t][l+1].at<double>(r-1, c-1)
//  ) || (
//    value >= img[t][l].at<double>(r+1, c+1) &&
//    value >= img[t][l].at<double>(r+1, c  ) &&
//    value >= img[t][l].at<double>(r+1, c-1) &&
//    value >= img[t][l].at<double>(r  , c+1) &&
//    value >= img[t][l].at<double>(r  , c-1) &&
//    value >= img[t][l].at<double>(r-1, c+1) &&
//    value >= img[t][l].at<double>(r-1, c  ) &&
//    value >= img[t][l].at<double>(r-1, c-1) &&
//    value >= img[t][l-1].at<double>(r+1, c+1) &&
//    value >= img[t][l-1].at<double>(r+1, c  ) &&
//    value >= img[t][l-1].at<double>(r+1, c-1) &&
//    value >= img[t][l-1].at<double>(r  , c+1) &&
//    value >= img[t][l-1].at<double>(r  , c  ) &&
//    value >= img[t][l-1].at<double>(r  , c-1) &&
//    value >= img[t][l-1].at<double>(r-1, c+1) &&
//    value >= img[t][l-1].at<double>(r-1, c  ) &&
//    value >= img[t][l-1].at<double>(r-1, c-1) &&
//    value >= img[t][l+1].at<double>(r+1, c+1) &&
//    value >= img[t][l+1].at<double>(r+1, c  ) &&
//    value >= img[t][l+1].at<double>(r+1, c-1) &&
//    value >= img[t][l+1].at<double>(r  , c+1) &&
//    value >= img[t][l+1].at<double>(r  , c  ) &&
//    value >= img[t][l+1].at<double>(r  , c-1) &&
//    value >= img[t][l+1].at<double>(r-1, c+1) &&
//    value >= img[t][l+1].at<double>(r-1, c  ) &&
//    value >= img[t][l+1].at<double>(r-1, c-1)  
//  ));
//}
