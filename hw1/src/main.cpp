#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

string all_hdr_type[] = {"Debevec"};
int valid_hdr_cnt[] = {2};
string all_tonemap_type[] = {"Reinhard"};
int valid_tonemap_cnt[] = {4};
string all_fusion_type[] = {"Mertens"};
int valid_fusion_cnt[] = {3};

string in_dir = "input_image";
string out_hdr = "result/out.hdr";
string out_jpg = "result/out.jpg";
int method = 1, hdr_type = 0, tonemap_type = 0, fusion_type = 0;
bool ghost = false, verbose = false;
vector<int> algn = {7, 4};
vector<double> hdr_para = {5, 60}, tonemap_para = {0, 0, 1, 0};
vector<double> fusion_para = {1, 1, 1};

#include "hdr.hpp"
#include "mtb.hpp"
#include "tonemap.hpp"
#include "util.hpp"

int main (int argc, char** argv) {
  int state = parse(argc, argv);
  if(!state) return 0;
  else if(state < 0) return 1;
  srand(time(NULL));
  namedWindow("show", WINDOW_NORMAL);

  in_dir += "/";
  ifstream ifs(in_dir+"input.txt", ifstream::in);
  int pic_num; ifs >> pic_num;
  //pic_num = 3; //while debugging
  vector<Mat> pics(pic_num);
  vector<double> etimes(pic_num);
  for(int i = 0; i<pic_num; ++i) {
    string pic_name; ifs >> pic_name >> etimes[i];
    pics[i] = imread(in_dir+pic_name, IMREAD_COLOR);
    CV_Assert(!pics[i].empty());
  }
  ifs.close();

  vector<Mat> aligned;
  if(algn[0] >= 0) {
    cerr << "start alignment...";
    MTB mtb(algn);
    mtb.process(pics, aligned);
    cerr << "done" << endl;
  } else {
    aligned = pics;
    cerr << "skip alignment" << endl;
  }
  
  vector<Mat> W;
  if(ghost) {
    cerr << "start ghost-removal..." << endl;
    ghost_removal(pics, W);
    cerr << "done" << endl;
  } else {
    cerr << "skip ghost-removal" << endl;
  }

  Mat ldr, hdr;
  if(!method) {
    cerr << "start hdr-ing...";
    DEBEVEC debevec(hdr_para);
    debevec.process(aligned, etimes, W, hdr);
    imwrite(out_hdr, hdr);
    cerr << "done" << endl;
  
    cerr << "start tonemapping...";
    TONEMAP tonemap(tonemap_para);
    tonemap.process(hdr, ldr);
    imwrite(out_jpg, ldr);
    cerr << "done" << endl;
  } else if(method == 1) {
    cerr << "start exposure fusion...";
    MERTENS mertens(fusion_para);
    mertens.process(aligned, W, ldr);
    imwrite(out_jpg, ldr*255);
    cerr << "done" << endl;
  }
  if(verbose) show(ldr);
}
