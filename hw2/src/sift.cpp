#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "sift.hpp"

#define OCTAVE_LAYER            5       
#define OCTAVE_NUM              3
#define OCTAVE_SCALE            3
#define GAUSSIAN_KERN           7
#define PRE_SIGMA               1.6
#define SIGMA                   pow(2, 1.0/(double)OCTAVE_SCALE)
#define CONTRAST_THRESHOLD      0.3
#define CURVATURE_THRESHOLD_R   10.0
#define CURVATURE_THRESHOLD     pow(CURVATURE_THRESHOLD_R+1, 2)/CURVATURE_THRESHOLD_R

inline bool is_extrema(const vector<vector<Mat>>&, int, int, int, int);

vector<Mat> SIFT::process(const vector<Mat>& img) {
  vector<Mat> L(img);

  #ifdef SHOW_PROCESS
  namedWindow("process", WINDOW_AUTOSIZE);
  #endif

  for (int i=0, n=img.size(); i<n; ++i) {
    // preprocessing images
    cvtColor(L[i], L[i], CV_BGR2GRAY);
    L[i].convertTo(L[i], CV_64FC1);
    L[i] *= 1./255;

    /**************************************/
    /** detection of scale-space extrema **/
    /**************************************/
    vector<vector<Mat>> g_octaves(OCTAVE_NUM);  // Gaussian octaves
    vector<vector<Mat>> d_octaves(OCTAVE_NUM);  // DoG octaves
    for (int t=0; t<OCTAVE_NUM; ++t) {
      g_octaves[t].resize(OCTAVE_LAYER);
      d_octaves[t].resize(OCTAVE_LAYER-1);
      resize(L[i], g_octaves[t][0], Size(), 2*pow(0.5,t), 2*pow(0.5,t));
      // compute Gaussian octaves
      for (int l=0; l<OCTAVE_LAYER; ++l) {
        double sigma = PRE_SIGMA * pow(SIGMA, l) * pow(2, t);
        GaussianBlur(
          g_octaves[t][0],
          g_octaves[t][l],
          Size((int)(sigma*4)*2+1, (int)(sigma*4)*2+1),
          sigma
        );
      }
      // comput DoG octaves
      for (int l=0; l<OCTAVE_LAYER-1; ++l)
        d_octaves[t][l] = g_octaves[t][l+1] - g_octaves[t][l];
    }

    /**************************************/
    /**  accurate keypoint localization  **/
    /**************************************/
    int lim = (int)(1+(1+2*GAUSSIAN_KERN*SIGMA)/5);
    for (int t=0; t<OCTAVE_NUM; ++t)
      for (int l=1, l_max=OCTAVE_LAYER-2; l<l_max; ++l) {

        #ifdef SHOW_PROCESS
        Mat marked_img = g_octaves[t][l];
        marked_img.convertTo(marked_img, CV_32FC1);
        cvtColor(marked_img, marked_img, CV_GRAY2BGR);
        #endif
        
        for (int c=lim, c_max=d_octaves[t][l].cols-lim; c<c_max; ++c)
          for (int r=lim, r_max=d_octaves[t][l].rows-lim; r<r_max; ++r) {
            if (!is_extrema(d_octaves, t, l, r, c)) continue;
            double value = d_octaves[t][l].at<double>(r, c);
            cerr << value << endl;
            // thow out low contrast
            double Dx = (
              d_octaves[t][l].at<double>(r, c+1) - 
              d_octaves[t][l].at<double>(r, c-1)
            ) / 2;
            double Dy = (
              d_octaves[t][l].at<double>(r+1, c) - 
              d_octaves[t][l].at<double>(r-1, c)
            ) / 2;
            double Ds = (
              d_octaves[t][l+1].at<double>(r, c) - 
              d_octaves[t][l-1].at<double>(r, c)
            ) / 2;
            double Dxx = (
              d_octaves[t][l].at<double>(r, c-1) + 
              d_octaves[t][l].at<double>(r, c+1) -
              d_octaves[t][l].at<double>(r, c  ) * 2
            );
            double Dyy = (
              d_octaves[t][l].at<double>(r-1, c) + 
              d_octaves[t][l].at<double>(r+1, c) -
              d_octaves[t][l].at<double>(r  , c) * 2
            );
            double Dss = (
              d_octaves[t][l-1].at<double>(r, c) + 
              d_octaves[t][l+1].at<double>(r, c) -
              d_octaves[t][l  ].at<double>(r, c) * 2
            );
            double Dxy = (
              d_octaves[t][l].at<double>(r-1, c-1) +
              d_octaves[t][l].at<double>(r+1, c+1) -
              d_octaves[t][l].at<double>(r-1, c+1) -
              d_octaves[t][l].at<double>(r+1, c-1)
            ) / 4;
            double Dxs = (
              d_octaves[t][l-1].at<double>(r, c-1) +
              d_octaves[t][l+1].at<double>(r, c+1) -
              d_octaves[t][l+1].at<double>(r, c-1) -
              d_octaves[t][l-1].at<double>(r, c+1)
            ) / 4;
            double Dys = (
              d_octaves[t][l-1].at<double>(r-1, c) +
              d_octaves[t][l+1].at<double>(r+1, c) -
              d_octaves[t][l+1].at<double>(r-1, c) -
              d_octaves[t][l-1].at<double>(r+1, c)
            ) / 4;
            Mat H = (Mat_<double>(3,3) << 
              Dxx, Dxy, Dxs, 
              Dxy, Dyy, Dys,
              Dxs, Dys, Dss
            );
            Mat parD = (Mat_<double>(3,1) << Dx, Dy, Ds);
            Mat x = (Mat_<double>(3,1) << c, r, SIGMA);
            Mat h = (-1) * H.inv() * parD;
            value = value + 0.5 * parD.dot(h);
            // if (value < 7) continue;
            cerr << value << endl;
            // eliminate edge responses; H: Hessian
            double TrH = Dxx + Dyy;
            double DetH = Dxx * Dyy - pow(Dxy, 2);
            /*if (DetH == 0) continue;
            if (TrH*TrH/DetH > CURVATURE_THRESHOLD) {
              cerr << "eliminate edge" << endl;
              continue;
            }*/

            #ifdef SHOW_PROCESS
            drawMarker(marked_img, Point(c, r), Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
            #endif
          }
        #ifdef SHOW_PROCESS
        imshow("process", marked_img);
        #endif
        waitKey(0);
      }
  }

    /**************************************/
    /**      orientation assignment      **/
    /**************************************/


    /**************************************/
    /**       keypoint descriptor        **/
    /**************************************/

}

// helper functions
inline bool is_extrema(const vector<vector<Mat>>& img, int t, int l, int r, int c) {
  double value = img[t][l].at<double>(r, c);
  /*Mat low = (Mat_<double>(3, 3) <<
    img[t][l-1].at<double>(r+1, c-1),
    img[t][l-1].at<double>(r+1, c  ),
    img[t][l-1].at<double>(r+1, c+1),
    img[t][l-1].at<double>(r  , c-1),
    img[t][l-1].at<double>(r  , c  ),
    img[t][l-1].at<double>(r  , c+1),
    img[t][l-1].at<double>(r-1, c-1),
    img[t][l-1].at<double>(r-1, c  ),
    img[t][l-1].at<double>(r-1, c+1)
  );

  Mat mid = (Mat_<double>(3, 3) <<
    img[t][l].at<double>(r+1, c-1),
    img[t][l].at<double>(r+1, c  ),
    img[t][l].at<double>(r+1, c+1),
    img[t][l].at<double>(r  , c-1),
    img[t][l].at<double>(r  , c  ),
    img[t][l].at<double>(r  , c+1),
    img[t][l].at<double>(r-1, c-1),
    img[t][l].at<double>(r-1, c  ),
    img[t][l].at<double>(r-1, c+1)
  );

  Mat high = (Mat_<double>(3, 3) <<
    img[t][l+1].at<double>(r+1, c-1),
    img[t][l+1].at<double>(r+1, c  ),
    img[t][l+1].at<double>(r+1, c+1),
    img[t][l+1].at<double>(r  , c-1),
    img[t][l+1].at<double>(r  , c  ),
    img[t][l+1].at<double>(r  , c+1),
    img[t][l+1].at<double>(r-1, c-1),
    img[t][l+1].at<double>(r-1, c  ),
    img[t][l+1].at<double>(r-1, c+1)
  );

  Mat tmp = (Mat_<double>(27, 1) <<
    img[t][l].at<double>(r+1, c+1),
    img[t][l].at<double>(r+1, c  ),
    img[t][l].at<double>(r+1, c-1),
    img[t][l].at<double>(r  , c+1),
    img[t][l].at<double>(r  , c  ),
    img[t][l].at<double>(r  , c-1),
    img[t][l].at<double>(r-1, c+1),
    img[t][l].at<double>(r-1, c  ),
    img[t][l].at<double>(r-1, c-1),
    img[t][l-1].at<double>(r+1, c+1),
    img[t][l-1].at<double>(r+1, c  ),
    img[t][l-1].at<double>(r+1, c-1),
    img[t][l-1].at<double>(r  , c+1),
    img[t][l-1].at<double>(r  , c  ),
    img[t][l-1].at<double>(r  , c-1),
    img[t][l-1].at<double>(r-1, c+1),
    img[t][l-1].at<double>(r-1, c  ),
    img[t][l-1].at<double>(r-1, c-1),
    img[t][l+1].at<double>(r+1, c+1),
    img[t][l+1].at<double>(r+1, c  ),
    img[t][l+1].at<double>(r+1, c-1),
    img[t][l+1].at<double>(r  , c+1),
    img[t][l+1].at<double>(r  , c  ),
    img[t][l+1].at<double>(r  , c-1),
    img[t][l+1].at<double>(r-1, c+1),
    img[t][l+1].at<double>(r-1, c  ),
    img[t][l+1].at<double>(r-1, c-1)
  );
  double min, max, minLoc, maxLoc;
  minMaxLoc(tmp, &min, &max);
  cerr << high << endl << mid << endl << low << endl;
  cerr << value 
       << " min:" << min
       << " max:" << max << endl;
  waitKey(0);*/
  
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