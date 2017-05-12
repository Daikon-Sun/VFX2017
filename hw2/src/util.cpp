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
extern double zoom;

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
       "List of all input images. (Image names should be seperated by any "
       "spaces.)")
      ("out_prefix,o", value<string>(&out_prefix)->default_value(out_prefix),
       "Output prefix of the panorama images (Images for will be named as: "
       "out_prefix0.jpg, out_prefix1.jpg...).")
      ("verbose,v", value<bool>()
       ->implicit_value(verbose, verbose?"True":"False")->composing(),
       "Visualize the final result.")
      ("zoom,z", value<double>(&zoom)->default_value(zoom),
       "Scale the image according to this value before processing to achieve "
       "faster result and use lesser memory. For example: a [6000x4000] image "
       "with zoom = 0.2 will become a [1200x800] image.")
      ("panorama,p",
       value<int>(&panorama_mode)->default_value(panorama_mode),
       "modes of generating panorama:\n"
       "  0: \tlinear ordering(n) (The program will stitch the images according"
              " to the order in \"in_list\" one after one from left to right\n"
       "  1: \tany ordering(n^2) (The program will automatically stitch "
              "the images in any order and recognise all scenes to produce one "
              "panorama for a single scene\n")

      ("detection,d",
       value<int>(&detection_mode)->default_value(detection_mode),
       "modes of feature detection:\n"
       "  0: \tMSOP (Multi-Scale Oriented Patches\n"
       "  1: \tSIFT (Scale Invariant Feature Transform\n\n")

      ("matching,m",
       value<int>(&matching_mode)->default_value(matching_mode),
       "modes of feature matching:\n"
       "  0: \texhaustive search\n"
       "  1: \tHAAR wavelet-based hashing\n"
       "  2: \tFLANN (Fast Library for Approximate Nearest Neighbors")
      ("matching_para",
       value< vector<double> >(&matching_para)->multitoken(),
       "Parameters of the chosen feature matching mode.\n"
       "  0: (0, maximum y-coordinate displacement for geometric constraint)\n"
       "  1: (0, bin number), (1, threshold between 1-NN / (avg 2-NN))\n"
              "(2, maximum y-coordinate displacement for geometric constraint)\n"
       "  2: NONE\n\n")
      
      ("projection,j",
       value<int>(&projection_mode)->default_value(projection_mode),
       "Types of projection:\n"
       "  0: \tnone\n"
       "  1: \tcylindrical")
      ("projection_para",
       value< vector<double> >(&projection_para)->multitoken(),
       "Parameters of the chosen projection type.\n"
       "  0: NONE\n"
       "  1: (0, focal length)\n\n")

      ("stitching,s",
       value<int>(&stitching_mode)->default_value(stitching_mode),
       "modes of image stitching:\n"
       "  0: \ttranslation\n"
       "  1: \ttranslation + estimate focal length (deprecated)\n"
       "  2: \ttranslation + rotation (deprecated)\n"
       "  3: \thomography\n"
       "  4: \tautomatic stitching")
      ("stitching_para", 
       value< vector<double> >(&stitching_para)->multitoken(),
       "Parameters of the chosen image stitching mode.\n"
       "  0: (0, number of rounds for RANSAC), (1, threshold of inliner/"
              "outlier)\n"
       "  3: (0, threshold of inliner/ouliear)\n"
       "  4: (0, threshold of inliner/ouliear), (1, estimated focal length "
              "(nonzero) for initialization of bundle adjustment)\n\n")

      ("blending,b",
       value<int>(&blending_mode)->default_value(blending_mode),
       "modes of blending:\n"
       "  0: \taverage (simple average of overlapping region)\n"
       "  1: \tmulti-band")
      ("blending_para",
       value< vector<double> >(&blending_para)->multitoken(),
       "Parameters of the chosen blending mode.\n"
       "  0: NONE\n"
       "  1: (0, number of bands)");

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
