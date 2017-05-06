#include <chrono>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

typedef vector<float> Para;

#include "panorama.hpp"

PANORAMA::PANORAMA(const string& in_list, const string& out_jpg, 
                   const int& detection_method,
                   const int& matching_method,
                   const int& projection_method,
                   const int& stitching_method,
                   const Para& matching_para,
                   const Para& projection_para,
                   const Para& stitching_para) 
                  : _detection_method(detection_method),
                    _matching_method(matching_method), 
                    _projection_method(projection_method), 
                    _stitching_method(stitching_method), 
                    _out_jpg(out_jpg),
                    DETECTION(_imgs, _keypoints),
                    MATCHING(matching_para, _keypoints, _match_pairs),
                    PROJECTION(projection_para, _imgs, _keypoints),
                    STITCHING(stitching_para, _imgs,
                              _keypoints, _match_pairs, _shift) {
  ifstream ifs(in_list, ifstream::in);
  string fname;
  while(ifs >> fname) {
    Mat tmp = imread(fname, IMREAD_COLOR);
    resize(tmp, tmp, Size(), 0.25, 0.25);
    _imgs.push_back(tmp.clone());
  }
};
void PANORAMA::process() {
  using std::chrono::steady_clock;
  steady_clock::time_point start, end;

  cerr << "start ";
  start = steady_clock::now();
  switch(_detection_method) {
     case 0: MSOP(); break;
    default: MSOP();
  }
  end = steady_clock::now();
  cerr << " " << duration_cast<seconds>(end-start).count() << " secs" << endl;

  cerr << "start ";
  switch(_matching_method) {
     case 1: HAAR(); break;
    default: HAAR();
  }
  end = steady_clock::now();
  cerr << " " << duration_cast<seconds>(end-start).count() << " secs" << endl;

  cerr << "start ";
  switch(_projection_method) {
     case 0: no_projection(); break;
     case 1: cylindrical(); break;
    default: cylindrical();
  }
  end = steady_clock::now();
  cerr << " " << duration_cast<seconds>(end-start).count() << " secs" << endl;

  //visualize();
  //cerr << "start ";
  switch(_stitching_method) {
     case 0: translation(); break;
     //case 1: focal_length(imgs, _keypoints, _match_pairs); break;
    default: translation(); break;
  }
}
void PANORAMA::visualize() {
  cerr << __func__ << endl;
  for(size_t pic = 0; pic+1<_keypoints.size(); ++pic) {
    const Scalar red = Scalar(0, 0, 255);
    Mat img0 = _imgs[pic].clone();
    Mat img1 = _imgs[pic+1].clone();
    for (const auto& p : _match_pairs[pic]) {
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
    for(const auto& p : _match_pairs[pic]) {
      const Keypoint& kp0 = _keypoints[pic][p.first];
      const Keypoint& kp1 = _keypoints[pic+1][p.second];
      line(show, Point(kp0.x, kp0.y), 
           Point(sz[0].width+kp1.x, kp1.y), red, 2, 8);
    }
    namedWindow("process", WINDOW_NORMAL);
    imshow("process", show);
    waitKey(0);
  }
}
