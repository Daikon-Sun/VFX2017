#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include "mtb.hpp"
#include "debevec.hpp"

void show(const Mat& m) {
  imshow("show", m);
  waitKey(0);
}
void onMouse(int evt, int x, int y, int flags, void* param) {
  if(evt == CV_EVENT_LBUTTONDOWN) {
    vector<Point>* pts = (vector<Point>*)param;
    pts->push_back(Point(x, y));
  }
}
int main (int argc, char* argv[]) {
  assert(argc == 5);
  namedWindow("show", WINDOW_NORMAL);

  string dir = string(argv[1])+"/";
  int max_level = atoi(argv[2]);
  int max_denoise = atoi(argv[3]);
  float lambda = atof(argv[4]);

  ifstream ifs(dir+"all_input.txt", ifstream::in);
  int pic_num; ifs >> pic_num; //total number of pictures to be aligned
  //pic_num = 3; //while debugging
  vector<Mat> pics(pic_num); //original image
  vector<double> etimes(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    string pic_name; ifs >> pic_name >> etimes[i];
    pics[i] = imread(dir+pic_name, IMREAD_COLOR);
    CV_Assert(!pics[i].empty());
  }
  ifs.close();

  ifs = ifstream(dir+"sample.txt", ifstream::in);
  int sam_num; ifs >> sam_num;
  vector<Point> sample_points(sam_num);
  for(int i = 0; i<sam_num; ++i)
    ifs >> sample_points[i].x >> sample_points[i].y;
  ifs.close();
}
