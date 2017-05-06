#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

using namespace boost::program_options;
using namespace cv;
using namespace std;

typedef vector<float> Para;
typedef vector<Para> Paras;

#include "panorama.hpp"

vector<string> all_detection = {"MSOP"};
vector<string> all_matching = {"exhaustive search", 
                               "HAAR wavelet-based hashing"};
vector<string> all_projection = {"none", "cylindrical"};
vector<string> all_stitching = {"translation", "focal-length"};

int detection_method = 0;

int matching_method = 1;
Paras all_matching_para = {{}, {10, 0.65, 50}};
vector<int> matching_cnt = {0, 3};

int projection_method = 1;
Paras all_projection_para = {{}, {10000}};
vector<int> projection_cnt = {0, 1};

int stitching_method = 0;
Paras all_stitching_para = {{4000, 25}, {4000, 25}};
vector<int> stitching_cnt = {2, 2};

string in_list = "input_images.txt";
string out_jpg = "result/out.jpg";
int detection = 0;
bool verbose = true;

int main(int argc, char** argv) 
{
  srand(time(NULL));

  int state = parse(argc, argv);
  if(!state) return 0;
  else if(state < 0) return 1;

  PANORAMA panorama(in_list, out_jpg,
                    detection_method,
                    matching_method,
                    projection_method,
                    stitching_method,
                    all_matching_para[matching_method],
                    all_projection_para[projection_method],
                    all_stitching_para[stitching_method]);
  panorama.process();

  return 0;
}
