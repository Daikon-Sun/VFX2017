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
int main (int argc, char* argv[]) {
  assert(argc == 11);
  srand(time(NULL));
  namedWindow("show", WINDOW_NORMAL);

  string in_dir = string(argv[1])+"/";
  int max_level = atoi(argv[2]);
  int max_denoise = atoi(argv[3]);
  double lambda = atof(argv[4]);
  double f = atof(argv[5]);
  double m = atof(argv[6]);
  double a = atof(argv[7]);
  double c = atof(argv[8]);
  string out_hdr_file = string(argv[9]);
  string out_jpg_file = string(argv[10]);

  ifstream ifs(in_dir+"input.txt", ifstream::in);
  int pic_num; ifs >> pic_num; //total number of pictures to be aligned
  //pic_num = 3; //while debugging
  vector<Mat> pics(pic_num); //original image
  vector<double> etimes(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    string pic_name; ifs >> pic_name >> etimes[i];
    pics[i] = imread(in_dir+pic_name, IMREAD_COLOR);
    CV_Assert(!pics[i].empty());
  }
  ifs.close();

  cerr << "start alignment...";
  MTB mtb(pics);
  vector<Mat> aligned;
  mtb.process(aligned, max_level, max_denoise);
  cerr << "done" << endl;
  
  Mat ldr, hdr;
  /*
  MERTENS mertens(aligned);
  mertens.process(ldr);
  show(ldr);
  imwrite(out_jpg_file, ldr*255);
  exit(0);
  */
  
  //vector<Mat> aligned = pics;
  cerr << "start hdr-ing...";
  DEBEVEC debevec(aligned, etimes);
  debevec.process(hdr, lambda);
  imwrite(out_hdr_file, hdr);
  cerr << "done" << endl;

  //ifstream par_in(in_dir+"tonemap_parameter.txt", ifstream::in);
  //double f, m, a, c;
  //par_in >> f >> m >> a >> c;
  TONEMAP tonemap(f, m, a ,c);
  tonemap.process(hdr, ldr);
  imwrite(out_jpg_file, ldr);
  show(ldr);
}
