#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

string all_hdr_type[] = {"Debevec"};
int valid_hdr_cnt[] = {2};
string all_tonemap_type[] = {"Reinhard"};
int valid_tonemap_cnt[] = {4};
string all_fusion_type[] = {"Mertens"};
int valid_fusion_cnt[] = {4};

string in_dir = "input_image";
string out_hdr = "result/out.hdr";
string out_jpg = "result/out.jpg";
int method = 1, hdr_type = 0, tonemap_type = 0, fusion_type = 0, ghost = 0;
bool verbose = false, blob = false, blob_tune = false;
vector<int> algn = {7, 4};
vector<double> hdr_para = {5, 60}, tonemap_para = {-0.5, 0, 1, 0};
vector<double> spotlight, fusion_para = {1, 1, 1, 8};

#include "hdr.hpp"
#include "mtb.hpp"
#include "tonemap.hpp"
#include "util.hpp"

int main (int argc, char** argv) {
  int state = parse(argc, argv);
  if(!state) return 0;
  else if(state < 0) return 1;
  srand(time(NULL));

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
  
  if(blob_tune) {
    cerr << "start tuing blob-removal parameters..." << endl;
    cerr << "press esc to continue" << endl;
    for(int i = 0; i<pic_num; ++i) tune_blob(pics[i]);
    cerr << "done" << endl;
    exit(0);
  } else {
    cerr << "skip blob-removal parameters tuning" << endl;
  }

  if(algn[0] >= 0) {
    cerr << "start alignment...";
    MTB mtb(algn);
    mtb.process(pics);
    cerr << "done" << endl;
  } else {
    cerr << "skip alignment" << endl;
  }
  
  if(blob) {
    cerr << "start blob-removal...";
    for(int i = 0; i<pic_num; ++i) blob_removal(pics[i], pics[i]);
    cerr << "done" << endl;
  } else {
    cerr << "skip blob-removal" << endl;
  }
  /*
  namedWindow("show", WINDOW_NORMAL);
  for(auto& p:pics) {
    imshow("show", p);
    waitKey(0);
  }
  */

  vector<Mat> W;
  if(ghost > 0) {
    cerr << "start ghost-removal...";
    ghost_removal(pics, ghost, W);
    cerr << "done" << endl;
  } else {
    cerr << "skip ghost-removal" << endl;
  }
   
  if(!spotlight.empty()) {
    cerr << "start spotlight-ing...";
    add_spotlight(pics, spotlight);
    cerr << "done" << endl;
  } else {
    cerr << "skip spotlight-ing" << endl;
  }

  Mat ldr, hdr;
  if(!method) {
    cerr << "start hdr-ing...";
    DEBEVEC debevec(hdr_para);
    debevec.process(pics, etimes, W, hdr);
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
    mertens.process(pics, W, ldr);
    imwrite(out_jpg, ldr*255);
    cerr << "done" << endl;
  }
  if(verbose) {
    namedWindow("show", WINDOW_NORMAL);
    imshow("show", ldr);
    waitKey(0);
    destroyWindow("show");
  }
}
