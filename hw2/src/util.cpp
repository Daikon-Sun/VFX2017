#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

using namespace boost::program_options;
using namespace cv;
using namespace std;

#include "util.hpp"

typedef vector<float> Para;
typedef vector<Para> Paras;

extern vector<string> all_detection, all_matching, all_projection;
extern vector<string> all_stitching;
extern Paras all_matching_para, all_projection_para, all_stitching_para;
extern int detection_method, matching_method, projection_method;
extern int stitching_method;
extern vector<int> matching_cnt, projection_cnt, stitching_cnt;
extern string in_list, out_jpg;
extern bool verbose;

bool check(const vector<float>& para, const string& name,
           const int& cnt, const string& info) {
  if((int)para.size() != cnt) {
    cout << info << " arguments: " << name << " should have " << cnt
         << " argument" << (cnt?"s!":"!") << endl;
    return false;
  }
  return true;
}
int parse(int ac, char** av) {
  Para matching_para, projection_para, stitching_para;
	options_description desc("All Available Options in VFX2017 hw2 project");
	desc.add_options()
			("help,h", "Print help message.\n")
      ("in_list,i", value<string>(&in_list)->default_value(in_list),
       "List of all input images.\n")
      ("out_jpg,o", value<string>(&out_jpg)->default_value(out_jpg),
       "Output filename of the panorama image.\n")

      ("detection,d",
       value<int>(&detection_method)->default_value(detection_method),
       "Methods of feature detection:\n"
       "  0: \tMSOP\n")

      ("matching,m",
       value<int>(&matching_method)->default_value(matching_method),
       "Methods of feature matching:\n"
       "  0: \texhaustive search\n"
       "  1: \tHAAR wavelet-based hashing\n")
      ("matching_para",
       value< vector<float> >(&matching_para)->multitoken(),
       "Parameters of the chosen feature matching method.\n")
      
      ("projection,p",
       value<int>(&projection_method)->default_value(projection_method),
       "Types of projection:\n"
       "  0: \tnone\n"
       "  1: \tcylindrical")
      ("projection_para",
       value< vector<float> >(&projection_para)->multitoken(),
       "Parameters of the chosen projection type.\n")

      ("stitching,s",
       value<int>(&stitching_method)->default_value(stitching_method),
       "Methods of image stitching:\n"
       "  0: \tRANSAC\n")
      ("stitching_para", 
       value< vector<float> >(&stitching_para)->multitoken(),
       "Parameters of the chosen image stitching method.\n")
      ("verbose,v", value<bool>()
       ->implicit_value(verbose, verbose?"True":"False")->composing(),
       "Show the final result.");

	variables_map vm;
	store(parse_command_line(ac, av, desc), vm);
	notify(vm);
	if(vm.count("help")) {
			cout << desc << endl;
			return 0;
	}
  if(vm.count("matching_para")) {
    if(!check(matching_para, all_matching[matching_method], 
              matching_cnt[matching_method], "matching_para"))
      return -1;
    all_matching_para[matching_method] = matching_para;
  }
  if(vm.count("projection_para")) {
     if(!check(projection_para, all_projection[projection_method], 
               projection_cnt[projection_method], "projection_para"))
      return -1;
    all_projection_para[projection_method] = projection_para;
  }
  if(vm.count("stitching_para")) {
    if(!check(stitching_para, all_stitching[stitching_method], 
              stitching_cnt[stitching_method], "stitching"))
      return -1;
    all_stitching_para[stitching_method] = stitching_para;
  }

  if(projection_para.empty() && stitching_method == 0)
    stitching_method = 1;

  verbose = vm.count("verbose");

  return 1;
}
