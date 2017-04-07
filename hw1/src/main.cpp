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
  assert(argc == 7);
  srand(time(NULL));
  namedWindow("show", WINDOW_NORMAL);

  string in_dir = string(argv[1])+"/";
  int max_level = atoi(argv[2]);
  int max_denoise = atoi(argv[3]);
  double lambda = atof(argv[4]);
  string out_hdr_file = string(argv[5]);
  string out_jpg_file = string(argv[6]);

  ifstream ifs(in_dir+"input.txt", ifstream::in);
  int pic_num; ifs >> pic_num; //total number of pictures to be aligned
  //pic_num = 3; //while debugging
  vector<Mat> pics(pic_num); //original image
  vector<double> etimes(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    string pic_name; ifs >> pic_name >> etimes[i];
    etimes[i] = 1.0/etimes[i];
    pics[i] = imread(in_dir+pic_name, IMREAD_COLOR);
    CV_Assert(!pics[i].empty());
  }
  ifs.close();

  MTB mtb(pics);
  vector<Mat> aligned;
  mtb.process(aligned, max_level, max_denoise);
  
  Mat ldr, hdr;
  MERTENS mertens(pics);
  mertens.process(ldr);
  exit(0);

  DEBEVEC debevec(pics, etimes);
  debevec.process(hdr, lambda);
  imwrite(out_hdr_file, hdr);

  ifstream par_in(in_dir+"tonemap_parameter.txt", ifstream::in);
  double f, m, a, c;
  par_in >> f >> m >> a >> c;
  TONEMAP tonemap(f, m, a ,c);
  tonemap.process(hdr, ldr);
  imwrite(out_jpg_file, ldr);
  //show(ldr);
}
