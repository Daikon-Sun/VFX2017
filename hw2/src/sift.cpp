#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "sift.hpp"

#define OCTAVE_LAYER      5       
#define OCTAVE_NUM        4
#define OCTAVE_SCALE      4
#define GAUSSIAN_KERN     3
#define SIGMA             pow(2, 1.0 / (double)OCTAVE_SCALE)

vector<Mat> SIFT::process(const vector<Mat>& img) {
  vector<Mat> gray_img(img);

  /**************************************/
  /** detection of scale-space extrema **/
  /**************************************/
  // build octaves
  for (int i=0, n=img.size(); i<n; ++i) {
    cvtColor(img[i], gray_img[i], COLOR_BGR2GRAY);
    vector<vector<Mat>> octaves(OCTAVE_NUM);

    for (int t=0; t<OCTAVE_NUM; ++t) {
      octaves[t].resize(OCTAVE_LAYER);
      // compute first layer of each octave
      if (t==0)
        GaussianBlur(
          gray_img[i], 
          octaves[t][0], 
          Size(GAUSSIAN_KERN, GAUSSIAN_KERN),
          SIGMA
        );
      else
        pyrDown(
          octaves[t-1][OCTAVE_SCALE-1],
          octaves[t][0]
        );
      // compute other layers of each octave
      for (int l = 1; l < OCTAVE_LAYER; ++l) 
        GaussianBlur(
          octaves[t][l-1],
          octaves[t][l],
          Size(GAUSSIAN_KERN, GAUSSIAN_KERN),
          SIGMA
        );
    }

    /*for(int t=0; t<OCTAVE_NUM; ++t)
      for(int l=0; l<OCTAVE_LAYER; ++l) {
        namedWindow("test", WINDOW_AUTOSIZE);
        imshow("test", octaves[t][l]);
        waitKey(0);
      }*/
  }





  // accurate keypoint localization
  // TODO

  // orientation assignment
  // TODO

  // keypoint descriptor  
  // TODO
}