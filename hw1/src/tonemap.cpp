#include "tonemap.hpp"

void TONEMAP::process(Mat& input, Mat& output) {
  output = input.clone();

  Mat gray_img; // for calculating luminance
  cvtColor(input, gray_img, COLOR_RGB2GRAY);

  double Cav[3];
  double Lav = mean(gray_img).val[0];
  double L, Lmin, Lmax, Ia, Ig, Il;

  minMaxLoc(gray_img, &Lmin, &Lmax);
  _m = (_m > 0) ? _m : 0.3 + 0.7 * pow((log(Lmax) - log(Lav)) / (log(Lmax) - log(Lmin)), 1.4);

  cout << _m << endl;

  vector<Mat> channels_img; // for calculating channel averages
  split(input, channels_img);
  for (int ch = 0; ch < 3; ++ch)
    Cav[ch] = mean(channels_img[ch]).val[0];

  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      L = gray_img.at<float>(i, j);

      for (int ch = 0; ch < 3; ++ch) {
        Il = _c * input.at<Vec3f>(i, j)[ch] + (1-_c) * L;
        Ig = _c * Cav[ch] + (1-_c) * Lav;
        Ia = _a * Il + (1-_a) * Ig;
        //cerr << input.at<Vec3f>(i, j)[ch] + pow(_f * Ia, _m) << endl;
        output.at<Vec3f>(i, j)[ch] /= input.at<Vec3f>(i, j)[ch] + pow(_f * Ia, _m);
      }
    }
  normalize(output, output, 255, 0);
}
