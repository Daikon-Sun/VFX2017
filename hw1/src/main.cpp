#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include "mtb.hpp"
#include "hdr.hpp"
#include "tonemap.hpp"

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
  assert(argc == 7);
  namedWindow("show", WINDOW_NORMAL);

  string in_dir = string(argv[1])+"/";
  int max_level = atoi(argv[2]);
  int max_denoise = atoi(argv[3]);
  float lambda = atof(argv[4]);
  string out_hdr_file = string(argv[5]);
  string out_jpg_file = string(argv[6]);

  ifstream ifs(in_dir+"input.txt", ifstream::in);
  int pic_num; ifs >> pic_num; //total number of pictures to be aligned
  //pic_num = 3; //while debugging
  vector<Mat> pics(pic_num); //original image
  vector<float> etimes(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    string pic_name; ifs >> pic_name >> etimes[i];
    pics[i] = imread(in_dir+pic_name, IMREAD_COLOR);
    CV_Assert(!pics[i].empty());
  }
  ifs.close();

  MTB mtb(pics);
  vector<Mat> aligned;
  mtb.process(aligned, max_level, max_denoise);

  ifs = ifstream(in_dir+"sample.txt", ifstream::in);
  int sam_num; ifs >> sam_num;
  vector<Point> points(sam_num);
  for(int i = 0; i<sam_num; ++i) ifs >> points[i].x >> points[i].y;
  ifs.close();

  Mat ldr, hdr;

  DEBEVEC debevec(aligned, etimes, points);
  debevec.process(hdr, lambda);
  imwrite(out_hdr_file, hdr);

  ifstream par_in(in_dir+"tonemap_parameter.txt", ifstream::in);
  double f, m, a, c;
  par_in >> f >> m >> a >> c;
  // tonemap(f, m, a ,c)
  TONEMAP tonemap(f, m, a ,c);
  tonemap.process(hdr, ldr);
  show(ldr);
}
