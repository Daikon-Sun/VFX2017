#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//#include "sift.hpp"
#include "msop.hpp"

int main(int argc, char** agrv) 
{
  Mat pic1 = imread("test1.JPG", IMREAD_COLOR);
  Mat pic2 = imread("test2.JPG", IMREAD_COLOR);
  Mat pic3 = imread("test3.JPG", IMREAD_COLOR);

  vector<Mat> vec;
  vec.push_back(pic1);
  vec.push_back(pic2);
  vec.push_back(pic3);
  MSOP msop;
  msop.process(vec);

  return 0;
}
