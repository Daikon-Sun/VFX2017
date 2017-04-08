#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#include "tonemap.hpp"

void TONEMAP::process(Mat& input, Mat& output) {
  output = input.clone();

  Mat L_img = Mat::zeros(input.size(), CV_32FC1); // for calculating luminance
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      L_img.at<float>(i, j) = (
        input.at<Vec3f>(i, j)[0] * 0.0721 +
        input.at<Vec3f>(i, j)[1] * 0.7154 +
        input.at<Vec3f>(i, j)[2] * 0.2125) / 3;
    }

  double Cav[3];
  double Lav = mean(L_img).val[0];
  double L, Lmin, Lmax, Ia, Ig, Il;

  minMaxLoc(L_img, &Lmin, &Lmax);
  _m = (_m > 0) ? _m :
    0.3 + 0.7 * pow((log(Lmax) - log(Lav)) / (log(Lmax) - log(Lmin)), 1.4);

  vector<Mat> channels_img; // for calculating channel averages
  split(input, channels_img);
  for (int ch = 0; ch < 3; ++ch)
    Cav[ch] = mean(channels_img[ch]).val[0];

  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      L = L_img.at<float>(i, j);

      for (int ch = 0; ch < 3; ++ch) {
        Il = _c * input.at<Vec3f>(i, j)[ch] + (1-_c) * L;
        Ig = _c * Cav[ch] + (1-_c) * Lav;
        Ia = _a * Il + (1-_a) * Ig;
        output.at<Vec3f>(i, j)[ch] /= 
          input.at<Vec3f>(i, j)[ch] + pow(_f * Ia, _m);
        output.at<Vec3f>(i, j)[ch] =
          (output.at<Vec3f>(i, j)[ch] - Lmin) / (Lmax - Lmin);
      }
    }

  normalize(output, output, 255, 0, NORM_MINMAX, CV_8UC3);
}
