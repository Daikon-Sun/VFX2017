#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//#include "sift.hpp"
#include "msop.hpp"

float F;

int main(int argc, char** argv) 
{
  F = atof(argv[1]);

  srand(time(NULL));
  Mat pic1 = imread("test1.JPG", IMREAD_COLOR);
  resize(pic1, pic1, Size(), 0.25, 0.25);
  Mat pic2 = imread("test2.JPG", IMREAD_COLOR);
  resize(pic2, pic2, Size(), 0.25, 0.25);

  //Mat pic1 = imread("adobe_panoramas/data/carmel/carmel-17.png", IMREAD_COLOR);
  //Mat pic2 = imread("adobe_panoramas/data/carmel/carmel-16.png", IMREAD_COLOR);

  vector<Mat> vec;
  vec.push_back(pic1);
  vec.push_back(pic2);
  //vec.push_back(pic3);
  MSOP msop;
  msop.process(vec);

  return 0;
}
