#include "mtb.hpp"

void MTB::process(vector<Mat>& res, int max_level, int max_denoise) {
  int pics_num = (int)_pics.size();
  _bi_pics.resize(pics_num); //0-or-255 image after thresholding
  _masks.resize(pics_num); //mask to de-noise
  for(int i = 0; i<pics_num; ++i) {
    _bi_pics[i].resize(max_level);
    _masks[i].resize(max_level);
    Mat pic = _pics[i].clone();
    transform_bi(pic, _bi_pics[i][0], _masks[i][0], max_denoise);
    for(int j = 1; j<max_level; ++j) {
      resize(pic, pic, Size(), 0.5, 0.5, INTER_NEAREST);
      transform_bi(pic, _bi_pics[i][j], _masks[i][j], max_denoise);
    }
  }
  vector< pair<int, int> > offsets(pics_num);
  for(int i = 1; i<pics_num; ++i) offsets[i] = align(i, 0, max_level);
  int minc, minr, maxc, maxr, cols = _pics[0].cols, rows = _pics[0].rows;
  minc = minr = maxc = maxr = 0;
  for(auto p:offsets) {
    minc = min(minc, p.first);
    minr = min(minr, p.second);
    maxc = max(maxc, p.first);
    maxr = max(maxr, p.second);
  }
  res.resize(pics_num);
  for(int i = 0; i<pics_num; ++i) {
    int Dc, Dr; tie(Dc, Dr) = offsets[i];
    int c1 = maxc-Dc, c2 = cols+minc-Dc;
    int r1 = maxr-Dr, r2 = rows+minr-Dr;
    Rect roi(c1, r1, c2-c1, r2-r1);
    res[i] = _pics[i](roi).clone();
  }
}
void MTB::transform_bi(const Mat& m, Mat& bi, Mat& de, int max_denoise) {
  cvtColor(m, bi, COLOR_BGR2GRAY);
  vector<uchar> all_vals;
  CV_Assert(bi.isContinuous());
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
void MTB::shift(Mat& m, Mat& dst, const pair<int, int>& diff) {
  const int& Dc = diff.first;
  const int& Dr = diff.second;
  int c1 = (Dc<0 ? -Dc : 0);
  int r1 = (Dr<0 ? -Dr : 0);
  int c2 = (Dc<0 ? m.cols-c1 : m.cols-Dc-c1);
  int r2 = (Dr<0 ? m.rows-r1 : m.rows-Dr-r1);
  dst = Mat::zeros(m.size(), m.type());
  m(Rect(c1, r1, c2, r2)).copyTo(dst(Rect(m.cols-c2-c1, m.rows-r2-r1, c2, r2)));
}
pair<int, int> MTB::align(const int j, int lev, const int max_level) {
  if(lev == max_level) return {0, 0};

  pair<int, int> diff = align(j, lev+1, max_level);

  Mat& fixed = _bi_pics[0][lev];
  Mat& moved = _bi_pics[j][lev];
  Mat& msk1 = _masks[0][lev];
  Mat& msk2 = _masks[j][lev];
 
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
