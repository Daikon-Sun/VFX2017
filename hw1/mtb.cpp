#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

constexpr int max_level = 7; //maximum level to be shrinked
constexpr int max_denoise = 4; //range for de-noise

void show(const Mat& m) {
  imshow("show", m);
  waitKey(0);
}

void transform_bi(const Mat& m, Mat& bi, Mat& de) {
  cvtColor(m, bi, COLOR_BGR2GRAY); //transform to grayscale
  vector<uchar> all_vals;
  assert(bi.isContinuous());
  all_vals.assign(bi.datastart, bi.dataend);
  nth_element(all_vals.begin(), all_vals.begin()+all_vals.size()/2,
              all_vals.end());
  int median = *(all_vals.begin()+all_vals.size()/2); 
  Mat upp, low;
  threshold(bi, upp, median+max_denoise, 1, THRESH_BINARY);
  threshold(bi, low, median-max_denoise, 1, THRESH_BINARY_INV);
  de = upp+low;
  threshold(bi, bi, median, 255, THRESH_BINARY);
}
void shift(Mat& m, Mat& dst, const pair<int, int>& diff) {
  const int& Dc = diff.first;
  const int& Dr = diff.second;
  int c1 = (Dc<0 ? -Dc : 0);
  int r1 = (Dr<0 ? -Dr : 0);
  int c2 = (Dc<0 ? m.cols-c1 : m.cols-Dc-c1);
  int r2 = (Dr<0 ? m.rows-r1 : m.rows-Dr-r1);
  dst = Mat(m.size(), m.type(), Scalar::all(0));
  m(Rect(c1, r1, c2, r2)).copyTo(dst(Rect(m.cols-c2-c1, m.rows-r2-r1, c2, r2)));
}

pair<int, int> align(vector< vector<Mat> >& bi_pics,
                     vector< vector<Mat> >& masks, const int j, int lev) {
  if(lev == max_level) return {0, 0};

  pair<int, int> diff = align(bi_pics, masks, j, lev+1);

  Mat& fixed = bi_pics[0][lev];
  Mat& moved = bi_pics[j][lev];
  Mat& msk1 = masks[0][lev];
  Mat& msk2 = masks[j][lev];
 
  int best = fixed.cols*fixed.rows, bestc = -1, bestr = -1;
  for(int dc = -1; dc<2; ++dc) for(int dr = -1; dr<2; ++dr) {
    Mat moved_out, msk1_out, msk2_out, res;
    int Dc = 2*diff.first+dc, Dr = 2*diff.second+dr;
    shift(moved, moved_out, {Dc, Dr});
    shift(msk1, msk1_out, {Dc, Dr});
    shift(msk2, msk2_out, {Dc, Dr});
    bitwise_and(msk1_out, msk2_out, msk1_out);
    bitwise_xor(fixed, moved_out, res, msk1_out);
    int cnt = countNonZero(res);
    if( cnt < best ) {
      best = cnt;
      bestc = dc;
      bestr = dr;
    }
  }
  return {2*diff.first+bestc, 2*diff.second+bestr};
}

int main (int argc, char* argv[]) {
  assert(argc == 2);
  namedWindow("show", WINDOW_NORMAL);

  string dir = string(argv[1])+"/";
  ifstream ifs(dir+"file_list", ifstream::in);
  int num; ifs >> num; //total number of pictures to be aligned
  //num = 2; //while debugging
  vector<String> pics_name(num);
  vector<Mat> pics(num); //original image
  vector< vector<Mat> > bi_pics(num); //0-or-255 image after thresholding
  vector< vector<Mat> > masks(num); //mask to de-noise
  for(int i = 0; i<num; ++i) {
    string tmp; ifs >> tmp;
    tmp = dir+tmp;
    pics_name[i] = String(tmp); //convert c++ string to opencv String 
    pics[i] = imread(tmp, IMREAD_COLOR);
    bi_pics[i].resize(max_level);
    masks[i].resize(max_level);
    transform_bi(pics[i], bi_pics[i][0], masks[i][0]);
    for(int j = 1; j<max_level; ++j) {
      resize(pics[i], pics[i], Size(), 0.5, 0.5, INTER_NEAREST);
      transform_bi(pics[i], bi_pics[i][j], masks[i][j]);
    }
  }

  vector< pair<int, int> > offsets(num);
  for(int i = 1; i<num; ++i) {
    offsets[i] = align(bi_pics, masks, i, 0);
    cerr << offsets[i].first << " " << offsets[i].second << endl;
  }
}
