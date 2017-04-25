#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "sift.hpp"

int main(int argc, char** agrv) 
{
  Mat pic1 = imread("prtn13.jpg", IMREAD_COLOR);
  Mat pic2 = imread("prtn12.jpg", IMREAD_COLOR);

  vector<Mat> vec;
  vec.push_back(pic1);
  SIFT sift;
  sift.process(vec);

  return 0;
}