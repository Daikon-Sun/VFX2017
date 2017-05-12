#include <chrono>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

typedef vector<double> Para;

#include "panorama.hpp"

PANORAMA::PANORAMA(const string& in_list, const string& out_prefix, 
                   const int& panorama_mode,
                   const int& detection_mode,
                   const int& matching_mode,
                   const int& projection_mode,
                   const int& stitching_mode,
                   const int& blending_mode,
                   const Para& matching_para,
                   const Para& projection_para,
                   const Para& stitching_para,
                   const bool& verbose)
                  : _panorama_mode(panorama_mode),
                    _detection_mode(detection_mode),
                    _matching_mode(matching_mode), 
                    _projection_mode(projection_mode), 
                    _stitching_mode(stitching_mode), 
                    _blending_mode(blending_mode),
                    _out_prefix(out_prefix),
                    _verbose(verbose),
                    DETECTION(_imgs, _keypoints),
                    MATCHING(panorama_mode, matching_para, _imgs, 
                             _keypoints, _match_pairs),
                    PROJECTION(projection_para, _imgs, _keypoints),
                    STITCHING(panorama_mode, stitching_para, _imgs,
                              _keypoints, _match_pairs, _shift, _order),
                    BLENDING(_imgs, _shift, _order, _outputs) {
  ifstream ifs(in_list, ifstream::in);
  string fname;
  while(ifs >> fname) {
    Mat tmp = imread(fname, IMREAD_COLOR);
    resize(tmp, tmp, Size(), 0.1, 0.1);
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
  vector<type2> matchings = {&MATCHING::exhaustive,
                             &MATCHING::HAAR,
                             &MATCHING::FLANN};
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
                              &STITCHING::rotation,
                              &STITCHING::homography,
                              &STITCHING::autostitch};
  execute<type4>(stitchings[_stitching_mode]);
  //blending
  typedef void(BLENDING::*type5)();
  vector<type5> blendings = {&BLENDING::average,
                             &BLENDING::multi_band};
  execute<type5>(blendings[_blending_mode]);

  if(_verbose) visualize();
  for(size_t i = 0; i<_outputs.size(); ++i)
    imwrite(_out_prefix+to_string(i)+".jpg", _outputs[i]);
}
void PANORAMA::visualize() {
  cerr << __func__ << endl;
  namedWindow("final result", WINDOW_NORMAL);
  for(auto& out : _outputs) {
    imshow("final result", out);
    waitKey(0);
  }
  //for(size_t pic = 0; pic+1<keypoints.size(); ++pic) {
  //   const auto red = Scalar(0, 0, 255);
  //   Mat img0 = imgs[pic].clone();
  //   Mat img1 = imgs[pic+1].clone();
  //   for (const auto& p : inners[pic][pic+1]) {
  //     const Keypoint& kp0 = keypoints[pic][p.first];
  //     const Keypoint& kp1 = keypoints[pic+1][p.second];
  //     drawMarker(img0, Point(kp0.x, kp0.y), red, MARKER_CROSS, 20, 2);
  //     drawMarker(img1, Point(kp1.x, kp1.y), red, MARKER_CROSS, 20, 2);
  //   }
  //   Size sz[2];
  //   for(size_t i = 0; i<2; ++i) sz[i] = imgs[pic+i].size();
  //   Mat show(sz[0].height, sz[0].width+sz[1].width, CV_8UC3);
  //   Mat left(show, Rect(0, 0, sz[0].width, sz[0].height));
  //   Mat right(show, Rect(sz[0].width, 0, sz[1].width, sz[1].height));
  //   img0.copyTo(left);
  //   img1.copyTo(right);
  //   for(const auto& p : inners[pic][pic+1]) {
  //     const Keypoint& kp0 = keypoints[pic][p.first];
  //     const Keypoint& kp1 = keypoints[pic+1][p.second];
  //     line(show, Point(kp0.x, kp0.y), 
  //          Point(sz[0].width+kp1.x, kp1.y), red, 2, 8);
  //   }
  //   namedWindow("process", WINDOW_NORMAL);
  //   imshow("process", show);
  //   waitKey(0);
  //}
}
