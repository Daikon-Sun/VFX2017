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
                   const Para& stitching_para,
                   const bool& verbose)
                  : _panorama_mode(panorama_mode),
                    _detection_mode(detection_mode),
                    _matching_mode(matching_mode), 
                    _projection_mode(projection_mode), 
                    _stitching_mode(stitching_mode), 
                    _out_jpg(out_jpg),
                    _verbose(verbose),
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
    resize(tmp, tmp, Size(), 0.2, 0.2);
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
                              &STITCHING::rotation,
                              &STITCHING::autostitch};
  execute<type4>(stitchings[_stitching_mode]);

  visualization();
}
void PANORAMA::visualization() {
  cerr << __func__ << endl;
  size_t pic_num = _imgs.size();
  if(!_panorama_mode) {
    if(_stitching_mode == 1) {
      double f = 0;
      for(size_t pic = 0; pic<pic_num; ++pic) {
        f += _shift[pic][pic+1].at<double>(0, 1) / _imgs.size();
        _shift[pic][pic+1].at<double>(0, 1) = 0.0;
      }
      set_focal_length(f); 
      cylindrical();
    }
    if(_stitching_mode <= 3) {
      for(size_t pic = 1; pic+1<pic_num; ++pic)
        _shift[pic][pic+1] *= _shift[pic-1][pic];
      //vector<vector<vector<Point2d>>> new_pos(pic_num);
      //#pragma omp parallel for
      //for(size_t pic = 1; pic<pic_num; ++pic) {
      //  new_pos[pic].resize(_imgs[pic].cols, vector<Point2d>(_imgs[pic].rows));
      //  for(int x = 0; x<_imgs[pic].cols; ++x)
      //    for(int y = 0; y<_imgs[pic].rows; ++y) {
      //      Mat pos = _shift[pic-1][pic] * (Mat_<double>(3, 1) << x, y, 1);
      //      const double& nx = pos.at<double>(0, 0);
      //      const double& ny = pos.at<double>(1, 0);
      //      new_pos[pic][x][y] = {nx, ny};
      //      #pragma critical
      //      mnx = min(mnx, nx);
      //      mxx = max(mxx, nx);
      //      mny = min(mny, ny);
      //      mxy = max(mxy, ny);
      //    }
      //} 
      Mat show = Mat::zeros(1200, 1200, CV_8UC3);
      Mat tmp = _imgs[0] / _imgs.size();
      tmp.copyTo(show(Rect(0, 0, _imgs[0].cols, _imgs[0].rows)));
      for(size_t pic = 1; pic<pic_num; ++pic) {
        Mat res;
        warpPerspective(_imgs[pic], res, _shift[pic-1][pic], Size(1200, 1200));
        cerr << res.size() << endl;
        show += res/_imgs.size();
      }
      namedWindow("visualize", WINDOW_NORMAL);
      imshow("visualize", show);
      imwrite(_out_jpg, show);
      waitKey(0);
    }
  } else cerr << "not in mode O(n)!" << endl;
}
