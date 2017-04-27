#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "sift.hpp"
#include "msop.hpp"

int main(int argc, char** agrv) 
{
  Mat pic1 = imread("test.JPG", IMREAD_COLOR);
  Mat pic2 = imread("grail02.jpg", IMREAD_COLOR);

  vector<Mat> vec;
  vec.push_back(pic1);
  MSOP msop;
  msop.process(vec);

  return 0;
}