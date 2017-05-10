#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

using namespace boost::program_options;
using namespace cv;
using namespace std;

#include "util.hpp"

typedef vector<double> Para;
typedef vector<Para> Paras;

extern vector<string> all_panorama, all_detection, all_matching, all_projection;
extern vector<string> all_stitching, all_blending;
extern Paras all_matching_para, all_projection_para, all_stitching_para;
extern Paras all_blending_para;
extern int panorama_mode, detection_mode, matching_mode;
extern int projection_mode, stitching_mode, blending_mode;
extern vector<int> matching_cnt, projection_cnt, stitching_cnt, blending_cnt;
extern string in_list, out_prefix;
extern bool verbose;

bool check(const vector<double>& para, const string& name,
           const int& cnt, const string& info) {
  if((int)para.size() != cnt) {
    cout << info << " arguments: " << name << " should have " << cnt
         << " argument" << (cnt>1?"s!":"!") << endl;
    return false;
  }
  return true;
}
int parse(int ac, char** av) {
  Para matching_para, projection_para, stitching_para, blending_para;
	options_description desc("All Available Options in VFX2017 hw2 project");
	desc.add_options()
			("help,h", "Print help message.")
      ("in_list,i", value<string>(&in_list)->default_value(in_list),
       "List of all input images.")
      ("out_prefix,o", value<string>(&out_prefix)->default_value(out_prefix),
       "Output prefix of the panorama images.")
      ("verbose,v", value<bool>()
       ->implicit_value(verbose, verbose?"True":"False")->composing(),
       "Show the final result.")
      ("panorama,p",
       value<int>(&panorama_mode)->default_value(panorama_mode),
       "modes of generating panorama:\n"
       "  0: \tlinear ordering(n)\n"
       "  1: \tany ordering(n^2)")
      ("detection,d",
       value<int>(&detection_mode)->default_value(detection_mode),
       "modes of feature detection:\n"
       "  0: \tMSOP\n"
       "  1: \tSIFT\n\n")

      ("matching,m",
       value<int>(&matching_mode)->default_value(matching_mode),
       "modes of feature matching:\n"
       "  0: \texhaustive search\n"
       "  1: \tHAAR wavelet-based hashing")
      ("matching_para",
       value< vector<double> >(&matching_para)->multitoken(),
       "Parameters of the chosen feature matching mode.\n\n")
      
      ("projection,j",
       value<int>(&projection_mode)->default_value(projection_mode),
       "Types of projection:\n"
       "  0: \tnone\n"
       "  1: \tcylindrical")
      ("projection_para",
       value< vector<double> >(&projection_para)->multitoken(),
       "Parameters of the chosen projection type.\n\n")

      ("stitching,s",
       value<int>(&stitching_mode)->default_value(stitching_mode),
       "modes of image stitching:\n"
       "  0: \ttranslation\n"
       "  1: \ttranslation + estimate focal length\n"
       "  2: \ttranslation + rotation\n"
       "  3: \thomography\n"
       "  4: \tautomatic stitching")
      ("stitching_para", 
       value< vector<double> >(&stitching_para)->multitoken(),
       "Parameters of the chosen image stitching mode.\n\n")

      ("blending,b",
       value<int>(&blending_mode)->default_value(blending_mode),
       "modes of blending:\n"
       "  0: \taverage\n"
       "  1: \tmulti-band");

	variables_map vm;
	store(parse_command_line(ac, av, desc), vm);
	notify(vm);
	if(vm.count("help")) {
			cout << desc << endl;
			return 0;
	}
  if(vm.count("matching_para")) {
    if(!check(matching_para, all_matching[matching_mode], 
              matching_cnt[matching_mode], "matching_para"))
      return -1;
    all_matching_para[matching_mode] = matching_para;
  }
  if(vm.count("projection_para")) {
     if(!check(projection_para, all_projection[projection_mode], 
               projection_cnt[projection_mode], "projection_para"))
      return -1;
    all_projection_para[projection_mode] = projection_para;
  }
  if(vm.count("stitching_para")) {
    if(!check(stitching_para, all_stitching[stitching_mode], 
              stitching_cnt[stitching_mode], "stitching"))
      return -1;
    all_stitching_para[stitching_mode] = stitching_para;
  }
  if(vm.count("blending_para")) {
    if(!check(blending_para, all_blending[blending_mode], 
              blending_cnt[blending_mode], "blending"))
      return -1;
    all_blending_para[blending_mode] = blending_para;
  }

  verbose |= vm.count("verbose");

  return 1;
}
