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
constexpr double G_KERN            = 0;
constexpr double SIGMA_P           = 1.0;  // pyramid smoothing
constexpr double SIGMA_I           = 1.5;  // integration scale
constexpr double SIGMA_D           = 1.0;  // derivative scale
constexpr double SIGMA_O           = 4.5;  // orientation scale
constexpr double HM_THRESHOLD      = 0.0005;
constexpr double ANMS_ROBUST_RATIO = 0.9;
constexpr int KEYPOINT_NUM        = 500;
constexpr double SAMPLE_SPACING    = 5;

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
    img.convertTo(img, CV_64FC1);
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
          double val = HM.at<double>(y,x);
          if (val < HM_THRESHOLD) continue;
          if ( 
            val < HM.at<double>(y+1,x+1) ||
            val < HM.at<double>(y+1,x  ) ||
            val < HM.at<double>(y+1,x-1) ||
            val < HM.at<double>(y  ,x+1) ||
            val < HM.at<double>(y  ,x-1) ||
            val < HM.at<double>(y-1,x+1) ||
            val < HM.at<double>(y-1,x  ) ||
            val < HM.at<double>(y-1,x-1)
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
        double dx = (
          HM.at<double>(p.y, p.x+1) -
          HM.at<double>(p.y, p.x-1)
        ) / 2;
        double dy = (
          HM.at<double>(p.y+1, p.x) -
          HM.at<double>(p.y-1, p.x)
        ) / 2;
        double dxx = (
          HM.at<double>(p.y, p.x+1) + 
          HM.at<double>(p.y, p.x-1) -
          HM.at<double>(p.y, p.x  ) * 2
        );
        double dyy = (
          HM.at<double>(p.y+1, p.x) + 
          HM.at<double>(p.y-1, p.x) -
          HM.at<double>(p.y  , p.x) * 2
        );
        double dxy = (
          HM.at<double>(p.y+1, p.x+1) + 
          HM.at<double>(p.y-1, p.x-1) -
          HM.at<double>(p.y+1, p.x-1) -
          HM.at<double>(p.y-1, p.x+1)
        ) / 4;
        Mat m1 = (Mat_<double>(2,2) << dxx, dxy, dxy, dyy);
        Mat m2 = (Mat_<double>(2,1) << dx, dy);
        Mat xm = m1.inv() * m2;
        double x = p.x - xm.at<double>(0,0);
        double y = p.y - xm.at<double>(0,1);
        double ux = 
          Px.at<double>(ceil(y) , ceil(x) ) * (floor(x) - x) * (floor(y) - y) +
          Px.at<double>(ceil(y) , floor(x)) * (x - ceil(x) ) * (floor(y) - y) +
          Px.at<double>(floor(y), ceil(x) ) * (floor(x) - x) * (y - ceil(y) ) +
          Px.at<double>(floor(y), floor(x)) * (x - ceil(x) ) * (y - ceil(y) );
        double uy = 
          Py.at<double>(ceil(y) , ceil(x) ) * (floor(x) - x) * (floor(y) - y) +
          Py.at<double>(ceil(y) , floor(x)) * (x - ceil(x) ) * (floor(y) - y) +
          Py.at<double>(floor(y), ceil(x) ) * (floor(x) - x) * (y - ceil(y) ) +
          Py.at<double>(floor(y), floor(x)) * (x - ceil(x) ) * (y - ceil(y) );
        kpts.emplace_back(x, y, atan2(uy,ux)*180/M_PI);
      }

      // compute feature descriptor
      for(; j<kpts.size(); ++j) {
        auto& p = kpts[j];
        Mat m, rot;
        GaussianBlur(pyr[lev+1], m, Size(G_KERN, G_KERN), SIGMA_P);
        rot = getRotationMatrix2D(Point(p.x/2, p.y/2), p.t, 1);
        warpAffine(m, m, rot, m.size());
        m.convertTo(m, CV_32FC1);
        getRectSubPix(m, Size(40, 40), Point(p.x/2, p.y/2), p.patch);
        resize(p.patch, p.patch, Size(), 0.2, 0.2);
        Scalar mean, sd;
        meanStdDev(p.patch, mean, sd);
        p.patch = (p.patch-mean[0])/sd[0];
        p.patch.convertTo(p.patch, CV_32FC1);
        p.patch = haar * p.patch * haar_T;
        p.patch.convertTo(p.patch, CV_64FC1);
        p.x *= pow(2, lev);
        p.y *= pow(2, lev);
      }
    }
  }
}

#define OCTAVE_NUM              3
#define OCTAVE_SCALE_NUM        3
#define SIGMA                   pow(2, 1.0/(double)OCTAVE_SCALE_NUM)
#define OCTAVE_LAYER            OCTAVE_SCALE_NUM+3
#define GAUSSIAN_KERN           7
#define PRE_SIGMA               1.6
#define CONTRAST_THRES          0.03
#define CURVATURE_THRES_R       10.0
#define CURVATURE_THRES         pow(CURVATURE_THRES_R+1, 2)/CURVATURE_THRES_R
#define ORIENT_WINDOW           17
#define HALF_ORIENT             (ORIENT_WINDOW-1)/2
#define DESC_WINDOW             16
#define HALF_DESC               DESC_WINDOW/2
#define DESC_SIGMA              0.5*DESC_WINDOW

inline bool is_extrema(const vector<vector<Mat>>&, int, int, int, int);

void DETECTION::SIFT() {
  cerr << __func__;
  assert(ORIENT_WINDOW%2 == 1);
  namedWindow("process", WINDOW_NORMAL);

  vector<Mat> L;
  for(const auto& img : imgs) L.push_back(img.clone());
  keypoints.resize(imgs.size());
  
  #pragma omp parallel for
  for(size_t i=0; i<L.size(); ++i) {
    vector<SIFTpoint> siftpoints;
    // preprocessing images
    cvtColor(L[i], L[i], CV_BGR2GRAY);
    L[i].convertTo(L[i], CV_64FC1);
    L[i] *= 1./255;
    /**************************************/
    /** detection of scale-space extrema **/
    /**************************************/
    vector< vector<Mat> > g_octaves(OCTAVE_NUM);  // Gaussian octaves
    vector< vector<Mat> > d_octaves(OCTAVE_NUM);  // DoG octaves
    #pragma omp parallel for
    for (int t=0; t<OCTAVE_NUM; ++t) {
      g_octaves[t].resize(OCTAVE_LAYER);
      d_octaves[t].resize(OCTAVE_LAYER-1);
      if(t == 0) g_octaves[t][0] = L[i].clone();
      else g_octaves[t][0] = g_octaves[t-1][OCTAVE_SCALE_NUM];
      // compute Gaussian octaves
      for (int l=1; l<OCTAVE_LAYER; ++l) {
        double sigma = PRE_SIGMA * pow(SIGMA, l-1);
        GaussianBlur(
          g_octaves[t][0],
          g_octaves[t][l],
          Size(),
          sigma, sigma
        );
      }
      // comput DoG octaves
      for (int l=0; l<OCTAVE_LAYER-1; ++l)
        d_octaves[t][l] = g_octaves[t][l+1] - g_octaves[t][l];
    }
    /**************************************/
    /**  accurate keypoint localization  **/
    /**************************************/
    //int lim = (int)(1+(1+2*GAUSSIAN_KERN*SIGMA)/5);
    const int lim = HALF_ORIENT+1;
    #pragma omp parallel for collapse(2)
    for (int t=0; t<OCTAVE_NUM; ++t)
      for (int l=1; l<OCTAVE_LAYER-2; ++l) {
        //#define SHOW_PROCESS
        #ifdef SHOW_PROCESS
        Mat marked_img = g_octaves[t][l];
        marked_img.convertTo(marked_img, CV_32FC1);
        cvtColor(marked_img, marked_img, CV_GRAY2BGR);
        marked_img.convertTo(marked_img, CV_64FC1);
        #endif
        
        for (int c=lim, c_max=d_octaves[t][l].cols-lim; c<c_max; ++c)
          for (int r=lim, r_max=d_octaves[t][l].rows-lim; r<r_max; ++r) {
            if (!is_extrema(d_octaves, t, l, r, c)) continue;
            bool found = false;
            int ll = l, cc = c, rr = r;
            while(!found) {
              double value = d_octaves[t][ll].at<double>(rr, cc);
              double Dx = (
                d_octaves[t][ll].at<double>(rr, cc+1) - 
                d_octaves[t][ll].at<double>(rr, cc-1)
              ) / 2;
              double Dy = (
                d_octaves[t][ll].at<double>(rr+1, cc) - 
                d_octaves[t][ll].at<double>(rr-1, cc)
              ) / 2;
              double Ds = (
                d_octaves[t][ll+1].at<double>(rr, cc) - 
                d_octaves[t][ll-1].at<double>(rr, cc)
              ) / 2;
              double Dxx = (
                d_octaves[t][ll].at<double>(rr, cc-1) + 
                d_octaves[t][ll].at<double>(rr, cc+1) -
                d_octaves[t][ll].at<double>(rr, cc  ) * 2
              );
              double Dyy = (
                d_octaves[t][ll].at<double>(rr-1, cc) + 
                d_octaves[t][ll].at<double>(rr+1, cc) -
                d_octaves[t][ll].at<double>(rr  , cc) * 2
              );
              double Dss = (
                d_octaves[t][ll-1].at<double>(rr, cc) + 
                d_octaves[t][ll+1].at<double>(rr, cc) -
                d_octaves[t][ll  ].at<double>(rr, cc) * 2
              );
              double Dxy = (
                d_octaves[t][ll].at<double>(rr-1, cc-1) +
                d_octaves[t][ll].at<double>(rr+1, cc+1) -
                d_octaves[t][ll].at<double>(rr-1, cc+1) -
                d_octaves[t][ll].at<double>(rr+1, cc-1)
              ) / 4;
              double Dxs = (
                d_octaves[t][ll-1].at<double>(rr, cc-1) +
                d_octaves[t][ll+1].at<double>(rr, cc+1) -
                d_octaves[t][ll+1].at<double>(rr, cc-1) -
                d_octaves[t][ll-1].at<double>(rr, cc+1)
              ) / 4;
              double Dys = (
                d_octaves[t][ll-1].at<double>(r-1, c) +
                d_octaves[t][ll+1].at<double>(r+1, c) -
                d_octaves[t][ll+1].at<double>(r-1, c) -
                d_octaves[t][ll-1].at<double>(r+1, c)
              ) / 4;
              Mat H = (Mat_<double>(3,3) << 
                Dxx, Dxy, Dxs, 
                Dxy, Dyy, Dys,
                Dxs, Dys, Dss
              );
              Mat parD = (Mat_<double>(3,1) << Dx, Dy, Ds);
              Mat h = (-1) * H.inv() * parD;
              found = true;
              Point pmn, pmx;
              double mn, mx;
              minMaxLoc(h, &mn, &mx, &pmn, &pmx);
              if(abs(mn) > 0.5 || abs(mx) > 0.5) {
                if(abs(h.at<double>(0, 0)) > 0.5)
                  cc += (h.at<double>(0, 0) < 0 ? -1 : 1);
                if(abs(h.at<double>(1, 0)) > 0.5)    
                  rr += (h.at<double>(0, 0) < 0 ? -1 : 1);
                if(abs(h.at<double>(2, 0)) > 0.5)    
                  ll += (h.at<double>(2, 0) < 0 ? -1 : 1);
                int good = 0;
                if(cc >= c_max) cc = c_max;
                else if(cc < lim) cc = lim;
                else ++good;
                if(rr >= r_max) rr = r_max;
                else if(rr < lim) rr = lim;
                else ++good;
                if(ll >= OCTAVE_LAYER-2) ll = OCTAVE_LAYER-2;
                else if(ll < 1) ll = 1;
                else ++good;
                if(good != 3) {
                  found = false;
                  break;
                }
              }
              if(!found) continue;
              double new_value = value + 0.5 * parD.dot(h);
              // thow out low contrast
              if(abs(new_value) <= CONTRAST_THRES) {
                found = false;
                break;
              }
              // eliminate edge responses; H: Hessian
              double TrH = Dxx + Dyy;
              double DetH = Dxx * Dyy - Dxy * Dxy;
              if(TrH * TrH / DetH >= CURVATURE_THRES) {
                found = false;
                break;
              }
            }
            if(found) {
              /**************************************/
              /**      orientation assignment      **/
              /**************************************/
              double sigma = PRE_SIGMA * pow(SIGMA, ll-1);
              Rect roi = Rect(cc-HALF_ORIENT, rr-HALF_ORIENT,
                              ORIENT_WINDOW, ORIENT_WINDOW);
              Mat G_W = L[i](Rect(roi)).clone();
              GaussianBlur(G_W, G_W, Size(), sigma*1.5, sigma*1.5);
              // Gaussian kernel Weight
              Mat MAG = Mat::zeros(ORIENT_WINDOW, ORIENT_WINDOW, CV_64FC1);
              double orient[ORIENT_WINDOW][ORIENT_WINDOW];
              for(int xx = cc-HALF_ORIENT; xx<cc+HALF_ORIENT; ++xx)
                for(int yy = rr-HALF_ORIENT; yy<rr+HALF_ORIENT; ++yy) {
                  double dx = g_octaves[t][ll].at<double>(yy, xx+1) -
                              g_octaves[t][ll].at<double>(yy, xx-1);
                  double dy = g_octaves[t][ll].at<double>(yy+1, xx) -
                              g_octaves[t][ll].at<double>(yy-1, xx);
                  int ny = yy-rr+HALF_ORIENT, nx = xx-cc+HALF_ORIENT;
                  MAG.at<double>(ny, nx) = sqrt(dx*dx + dy*dy);
                  orient[ny][nx] = atan2(dy, dx) * 180 / M_PI + 180;
                }
              // weighted by Gaussian kernel
              MAG = MAG.mul(G_W);
              // orientation assignment with histogram
              vector<double> bins(38);
              for(int nx = 0; nx<2*HALF_ORIENT; ++nx)
                for(int ny = 0; ny<2*HALF_ORIENT; ++ny)
                  bins[int(orient[ny][nx]/10-1e-20)+1] += MAG.at<double>(ny, nx);
              bins[0] = bins[36], bins[37] = bins[1];
              int mx1i = max_element(bins.begin()+1, bins.end()-1)-bins.begin();
              double better_mx1i = (bins[mx1i+1]-bins[mx1i-1]+bins[mx1i]) / 
                                    2 / bins[mx1i] + mx1i - 1;
              siftpoints.emplace_back(cc, rr, ll, better_mx1i*10+5, t);
              double mx1 = bins[mx1i];
              // consider all the rest w.r.t. the maximum peak
              for(int mx2i = 1; mx2i < 37; ++mx2i) {
                double mx2 = bins[mx2i];
                if(mx2i != mx1i && mx2 >= mx1*0.8) {
                  double better_mx2i = (bins[mx2i+1]-bins[mx2i-1]+bins[mx2i]) / 
                                        2 / bins[mx2i] + mx2i - 1;
                  siftpoints.emplace_back(cc, rr, ll, better_mx2i*10+5, t);
                }
              }
              #ifdef SHOW_PROCESS
              drawMarker(marked_img, Point(cc, rr),
                         Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
              #endif
            }
          }
        #ifdef SHOW_PROCESS
        imshow("process", marked_img);
        waitKey(0);
        #endif
      }
    /**************************************/
    /**       keypoint descriptor        **/
    /**************************************/
    keypoints[i].reserve(siftpoints.size()); 
    #pragma omp parallel for
    for(size_t j = 0; j<siftpoints.size(); ++j) {
      auto& spt = siftpoints[j];
      const int& midx = spt.x, midy = spt.y;
      keypoints[i].emplace_back(midx, midy, spt.t);
      Rect roi = Rect(midx-HALF_DESC, midy-HALF_DESC, DESC_WINDOW, DESC_WINDOW);
      Mat G_W = L[i](Rect(roi)).clone();
      GaussianBlur(G_W, G_W, Size(), DESC_SIGMA, DESC_SIGMA);
      double orient[DESC_WINDOW][DESC_WINDOW];
      Mat MAG = Mat::zeros(DESC_WINDOW, DESC_WINDOW, CV_64FC1);
      for(int x = midx-HALF_DESC; x<midx+HALF_DESC; ++x)
        for(int y = midy-HALF_DESC; y<midy+HALF_DESC; ++y) {
          double dx = g_octaves[spt.oc][spt.l].at<double>(y, x+1) -
                      g_octaves[spt.oc][spt.l].at<double>(y, x-1);
          double dy = g_octaves[spt.oc][spt.l].at<double>(y+1, x) -
                      g_octaves[spt.oc][spt.l].at<double>(y-1, x);
          int ny = y-midy+HALF_DESC, nx = x-midx+HALF_DESC;
          MAG.at<double>(ny, nx) = sqrt(dx*dx + dy*dy);
          orient[ny][nx] = atan2(dy, dx) * 180 / M_PI + 180 - spt.t;
          if(orient[ny][nx] >= 360) orient[ny][nx] -= 360;
          if(orient[ny][nx] < 0) orient[ny][nx] += 360;
        }
      MAG = MAG.mul(G_W);
      auto& kpt = keypoints[i].back();
      kpt.patch = Mat::zeros(128, 1, CV_64FC1);
      for(int lx = midx-HALF_DESC, cntx = 0; cntx<4; lx+=4, ++cntx)
        for(int ly = midy-HALF_DESC, cnty = 0; cnty<4; ly+=4, ++cnty)
          for(int x = lx; x<lx+4; ++x)
            for(int y = ly; y<ly+4; ++y) {
              int ny = y-midy+HALF_DESC, nx = x-midx+HALF_DESC;
              int ori = int(orient[ny][nx]/45-1e-20);
              kpt.patch.at<double>(cntx*32 + cnty*8 + ori, 0) 
                += MAG.at<double>(ny, nx);
            }
    }
  }
}
// helper functions
inline bool is_extrema(const vector<vector<Mat>>& img,
                       int t, int l, int r, int c) {
  double value = img[t][l].at<double>(r, c);
  return ((
    value <= img[t][l].at<double>(r+1, c+1) &&
    value <= img[t][l].at<double>(r+1, c  ) &&
    value <= img[t][l].at<double>(r+1, c-1) &&
    value <= img[t][l].at<double>(r  , c+1) &&
    value <= img[t][l].at<double>(r  , c-1) &&
    value <= img[t][l].at<double>(r-1, c+1) &&
    value <= img[t][l].at<double>(r-1, c  ) &&
    value <= img[t][l].at<double>(r-1, c-1) &&
    value <= img[t][l-1].at<double>(r+1, c+1) &&
    value <= img[t][l-1].at<double>(r+1, c  ) &&
    value <= img[t][l-1].at<double>(r+1, c-1) &&
    value <= img[t][l-1].at<double>(r  , c+1) &&
    value <= img[t][l-1].at<double>(r  , c  ) &&
    value <= img[t][l-1].at<double>(r  , c-1) &&
    value <= img[t][l-1].at<double>(r-1, c+1) &&
    value <= img[t][l-1].at<double>(r-1, c  ) &&
    value <= img[t][l-1].at<double>(r-1, c-1) &&
    value <= img[t][l+1].at<double>(r+1, c+1) &&
    value <= img[t][l+1].at<double>(r+1, c  ) &&
    value <= img[t][l+1].at<double>(r+1, c-1) &&
    value <= img[t][l+1].at<double>(r  , c+1) &&
    value <= img[t][l+1].at<double>(r  , c  ) &&
    value <= img[t][l+1].at<double>(r  , c-1) &&
    value <= img[t][l+1].at<double>(r-1, c+1) &&
    value <= img[t][l+1].at<double>(r-1, c  ) &&
    value <= img[t][l+1].at<double>(r-1, c-1)
  ) || (
    value >= img[t][l].at<double>(r+1, c+1) &&
    value >= img[t][l].at<double>(r+1, c  ) &&
    value >= img[t][l].at<double>(r+1, c-1) &&
    value >= img[t][l].at<double>(r  , c+1) &&
    value >= img[t][l].at<double>(r  , c-1) &&
    value >= img[t][l].at<double>(r-1, c+1) &&
    value >= img[t][l].at<double>(r-1, c  ) &&
    value >= img[t][l].at<double>(r-1, c-1) &&
    value >= img[t][l-1].at<double>(r+1, c+1) &&
    value >= img[t][l-1].at<double>(r+1, c  ) &&
    value >= img[t][l-1].at<double>(r+1, c-1) &&
    value >= img[t][l-1].at<double>(r  , c+1) &&
    value >= img[t][l-1].at<double>(r  , c  ) &&
    value >= img[t][l-1].at<double>(r  , c-1) &&
    value >= img[t][l-1].at<double>(r-1, c+1) &&
    value >= img[t][l-1].at<double>(r-1, c  ) &&
    value >= img[t][l-1].at<double>(r-1, c-1) &&
    value >= img[t][l+1].at<double>(r+1, c+1) &&
    value >= img[t][l+1].at<double>(r+1, c  ) &&
    value >= img[t][l+1].at<double>(r+1, c-1) &&
    value >= img[t][l+1].at<double>(r  , c+1) &&
    value >= img[t][l+1].at<double>(r  , c  ) &&
    value >= img[t][l+1].at<double>(r  , c-1) &&
    value >= img[t][l+1].at<double>(r-1, c+1) &&
    value >= img[t][l+1].at<double>(r-1, c  ) &&
    value >= img[t][l+1].at<double>(r-1, c-1)  
  ));
}
