#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

using namespace boost::program_options;
using namespace cv;
using namespace std;

typedef vector<double> Para;
typedef vector<Para> Paras;

#include "panorama.hpp"

vector<string> all_panorama = {"O(n)", "O(n^2)"};
vector<string> all_detection = {"MSOP", "SIFT"};
vector<string> all_matching = {"exhaustive search", 
                               "HAAR wavelet-based hashing", "FLANN"};
vector<string> all_projection = {"none", "cylindrical"};
vector<string> all_stitching = {"translation", "focal-length", "rotation",
                                "homography", "autostitch"};
vector<string> all_blending = {"average", "multi-band"};

int panorama_mode = 1;
int detection_mode = 1;

int matching_mode = 2;
Paras all_matching_para = {{50}, {10, 0.65, 50}, {}};
vector<int> matching_cnt = {0, 3, 0};

int projection_mode = 1;
Paras all_projection_para = {{}, {1500}};
vector<int> projection_cnt = {0, 1};

int stitching_mode = 4;
Paras all_stitching_para = {{1000, 2}, {1000, 2}, {1000, 2}, {10}, {5, 2000}};
vector<int> stitching_cnt = {2, 2, 2, 1, 2};

int blending_mode = 1;
Paras all_blending_para = {{}, {3}};
vector<int> blending_cnt = {0, 1};

string in_list = "input_images.txt";
string out_prefix = "result/out";
int detection = 0;
bool verbose = true;
double zoom = 0.3;

int main(int argc, char** argv) {
  srand(time(NULL));

  int state = parse(argc, argv);
  if(!state) return 0;
  else if(state < 0) return 1;

  PANORAMA panorama(in_list, out_prefix, zoom,
                    panorama_mode,
                    detection_mode,
                    matching_mode,
                    projection_mode,
                    stitching_mode,
                    blending_mode,
                    all_matching_para[matching_mode],
                    all_projection_para[projection_mode],
                    all_stitching_para[stitching_mode],
                    all_blending_para[blending_mode],
                    verbose);
  panorama.process();

  return 0;
}
