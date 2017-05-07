#include <chrono>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

typedef vector<double> Para;

#include "panorama.hpp"

PANORAMA::PANORAMA(const string& in_list, const string& out_jpg, 
                   const int& panorama_mode,
                   const int& detection_mode,
                   const int& matching_mode,
                   const int& projection_mode,
                   const int& stitching_mode,
                   const Para& matching_para,
                   const Para& projection_para,
                   const Para& stitching_para) 
                  : _panorama_mode(panorama_mode),
                    _detection_mode(detection_mode),
                    _matching_mode(matching_mode), 
                    _projection_mode(projection_mode), 
                    _stitching_mode(stitching_mode), 
                    _out_jpg(out_jpg),
                    DETECTION(_imgs, _keypoints),
                    MATCHING(panorama_mode, matching_para, _imgs, 
                             _keypoints, _match_pairs),
                    PROJECTION(projection_para, _imgs, _keypoints),
                    STITCHING(panorama_mode, stitching_para, _imgs,
                              _keypoints, _match_pairs, _shift) {
  ifstream ifs(in_list, ifstream::in);
  string fname;
  while(ifs >> fname) {
    Mat tmp = imread(fname, IMREAD_COLOR);
    resize(tmp, tmp, Size(), 0.25, 0.25);
    _imgs.push_back(tmp.clone());
  }
};
template<typename T>
void PANORAMA::execute(const T& f) {
  using std::chrono::steady_clock;
  steady_clock::time_point st, en;
  cerr << "start ";
  st = steady_clock::now();
  (this->*f)();
  en = steady_clock::now();
  cerr << " " << duration_cast<milliseconds>(en-st).count()/1000.0 
       << " secs" << endl;
}
void PANORAMA::process() {
  //feature detection
  typedef void (DETECTION::*type1)();
  vector<type1> detections = {&DETECTION::MSOP, &DETECTION::SIFT};
  execute<type1>(detections[_detection_mode]);
  //feature matching
  typedef void (MATCHING::*type2)();
  vector<type2> matchings = {&MATCHING::exhaustive, &MATCHING::HAAR};
  execute<type2>(matchings[_matching_mode]);
  //projection of both images and features
  typedef void(PROJECTION::*type3)();
  vector<type3> projections = {&PROJECTION::no_projection, 
                               &PROJECTION::cylindrical};
  execute<type3>(projections[_projection_mode]);
  //image stitching
  typedef void(STITCHING::*type4)();
  vector<type4> stitchings = {&STITCHING::translation,
                              &STITCHING::focal_length,
                              &STITCHING::rotation};
  execute<type4>(stitchings[_stitching_mode]);
}
void PANORAMA::visualize() {
  if(!_panorama_mode) {
    cerr << __func__ << endl;
    for(size_t pic = 0; pic+1<_keypoints.size(); ++pic) {
      const Scalar red = Scalar(0, 0, 255);
      Mat img0 = _imgs[pic].clone();
      Mat img1 = _imgs[pic+1].clone();
      for (const auto& p : _match_pairs[pic][pic+1]) {
        const Keypoint& kp0 = _keypoints[pic][p.first];
        const Keypoint& kp1 = _keypoints[pic+1][p.second];
        drawMarker(img0, Point(kp0.x, kp0.y), red, MARKER_CROSS, 20, 2);
        drawMarker(img1, Point(kp1.x, kp1.y), red, MARKER_CROSS, 20, 2);
      }
      Size sz[2];
      for(size_t i = 0; i<2; ++i) sz[i] = _imgs[pic+i].size();
      Mat show(sz[0].height, sz[0].width+sz[1].width, CV_8UC3);
      Mat left(show, Rect(0, 0, sz[0].width, sz[0].height));
      Mat right(show, Rect(sz[0].width, 0, sz[1].width, sz[1].height));
      img0.copyTo(left);
      img1.copyTo(right);
      for(const auto& p : _match_pairs[pic][pic+1]) {
        const Keypoint& kp0 = _keypoints[pic][p.first];
        const Keypoint& kp1 = _keypoints[pic+1][p.second];
        line(show, Point(kp0.x, kp0.y), 
             Point(sz[0].width+kp1.x, kp1.y), red, 2, 8);
      }
      namedWindow("visualize", WINDOW_NORMAL);
      imshow("visualize", show);
      waitKey(0);
    }
  } else cerr << "not in mode O(n)!" << endl;
}
