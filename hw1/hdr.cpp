#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/plot.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

using namespace cv;
using namespace plot;
using namespace std;

constexpr float lambda = 1;

void show(const Mat& m) {
  imshow("show", m);
  waitKey(0);
}
void onMouse(int evt, int x, int y, int flags, void* param) {
  if(evt == CV_EVENT_LBUTTONDOWN) {
    vector<Point>* pts = (vector<Point>*)param;
    pts->push_back(Point(x, y));
    cerr << x << " " << y << endl;
  }
}
inline int W(int val) {
  if(val <= 127) return val+1;
  else return 256-val;
}
int main (int argc, char* argv[]) {
  assert(argc == 2);
  namedWindow("show", WINDOW_NORMAL);

  string dir = string(argv[1])+"/";
  ifstream ifs(dir+"all_input.txt", ifstream::in);
  int pic_num; ifs >> pic_num; //total number of pictures to be aligned
  //pic_num = 3; //while debugging
  vector<String> pics_name(pic_num);
  vector<Mat> pics(pic_num); //original image
  vector<double> etimes(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    string tmp; ifs >> tmp >> etimes[i];
    tmp = dir+tmp;
    pics_name[i] = String(tmp); //convert c++ string to opencv String 
    pics[i] = imread(tmp, IMREAD_COLOR);
    CV_Assert(!pics[i].empty());
  }
  ifs.close();

  ifs = ifstream(dir+"sample.txt", ifstream::in);
  int sam_num; ifs >> sam_num;
  vector<Point> sample_points(sam_num);
  for(int i = 0; i<sam_num; ++i)
    ifs >> sample_points[i].x >> sample_points[i].y;
  ifs.close();
  int num_channels = pics[0].channels();
  vector<Mat> results(num_channels);
  for(int i = 0; i<num_channels; ++i) {
    Mat A = Mat::zeros(sam_num*pic_num + 257, 256 + sam_num, CV_64F);
    Mat B = Mat::zeros(A.rows, 1, CV_64F);

    int eq = 0;
    for(int j = 0; j<sam_num; ++j) for(int k = 0; k<pic_num; ++k) {
      int val = pics[k].at<Vec3b>(sample_points[j])[i];
      int w = W(val);
      A.at<double>(eq, val) = w;
      A.at<double>(eq, 256+j) = -w;
      B.at<double>(eq, 0) = w * log(etimes[k]);
      ++eq;
    }
    A.at<double>(eq, 128) = 1;
    ++eq;

    for(int j = 1; j<255; ++j) {
      int w = W(j);
      A.at<double>(eq, j-1) = lambda * w;
      A.at<double>(eq, j) = -2 * lambda * w;
      A.at<double>(eq, j+1) = lambda * w;
      ++eq;
    }
    solve(A, B, results[i], DECOMP_SVD);
    Ptr<Plot2d> plot = createPlot2d(-results[i].rowRange(0, 256));
    Mat display; plot->render(display);
    show(display);
  }
  //setMouseCallback("show", onMouse, (void*)&points); 
}
