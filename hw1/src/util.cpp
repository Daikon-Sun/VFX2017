#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace boost::program_options;

#include "util.hpp"

typedef Vec<double, 5> Vec5d;
constexpr double _2pi = pow(2*acos(-1), -2.5);

extern string all_hdr_type[];
extern int valid_hdr_cnt[];
extern string all_tonemap_type[];
extern int valid_tonemap_cnt[];
extern string all_fusion_type[];
extern int valid_fusion_cnt[];
extern string in_dir, out_hdr, out_jpg;
extern int method, hdr_type, tonemap_type, fusion_type;
extern bool ghost, verbose;
extern vector<int> algn;
extern vector<double> hdr_para, tonemap_para, fusion_para;

bool check(string str, size_t sz, int i) {
  if((int)sz != i) {
    cout << str+" should have "+to_string(i)+" argument"+(i>0?"s!":"!") << endl;
    return false;
  }
  return true;
}
int parse(int ac, char** av) {
  vector<double> tt;
	options_description desc("All Available Options in VFX2017 hw1 project");
	desc.add_options()
			("help,h", "Print help message.")
      ("in_dir,i", value<string>(&in_dir)->default_value(in_dir),
       "Input directory (all pictures and input.txt should be under it).")
      ("out_hdr_file,o", value<string>(&out_hdr)->default_value(out_hdr),
       "Output filename of hdr (including .hdr).")
      ("out_jpg_file,j", value<string>(&out_jpg)->default_value(out_jpg),
       "Output filename of jpg (including .jpg).")
      ("align,a",
       value< vector<int> >(&algn)->multitoken()->default_value(algn, "7 4"),
       "Align images before processing.")
      ("ghost,g", value<bool>()
       ->implicit_value(ghost, ghost?"True":"False")->composing(),
       "Add ghost-removal mask.")
      ("verbose,v", value<bool>()
       ->implicit_value(verbose, verbose?"True":"False")->composing(),
       "Show the final result.")
      ("method,m", value<int>(&method)->default_value(method),
       "Method to produce high-dynamic range image:\n"
       "  0: \thdr\n"
       "  1: \texposure fusion\n")
      ("hdr_type", value<int>(&hdr_type)->default_value(hdr_type),
       "Type of hdr:\n"
       "  0: \tDebevec")
      ("hdr_para", value< vector<double> >(&hdr_para)->multitoken(),
       "Parameters for the chosen hdr algorithm.\n")

      ("tonemape_type", value<int>(&tonemap_type)->default_value(tonemap_type),
       "Type of tonemap:\n"
       "  0:\tReinhard")
      ("tonemap_para", value< vector<double> >(&tonemap_para)->multitoken(),
       "Parameters for the chosen tonemapping algorithm.\n")

      ("fusion_type", value<int>(&fusion_type)->default_value(fusion_type),
       "Type of exposure fusion:\n"
       "  0:\tMertens")
      ("fusion_para", value< vector<double> >(&fusion_para)->multitoken(),
       "Parameters for the chosen exposure fusion algorithm.");

	variables_map vm;
	store(parse_command_line(ac, av, desc), vm);
	notify(vm);    

	if(vm.count("help")) {
			cout << desc << endl;
			return 0;
	}
  if(!algn.empty()) {
    if(!check("Aligment", algn.size(), 2)) return -1;
  }
  if(!method) {
    if(!check(all_hdr_type[hdr_type], hdr_para.size(), valid_hdr_cnt[hdr_type]))
      return -1;
    if(!check(all_tonemap_type[tonemap_type], tonemap_para.size(),
                      valid_tonemap_cnt[tonemap_type]))
      return -1;
  } else if(method == 1) {
    if(!check(all_fusion_type[fusion_type], fusion_para.size(),
                      valid_fusion_cnt[fusion_type]))
      return -1;
  } else {
    cout << "Method unavailable!" << endl;
    return -1;
  }
  verbose = vm.count("verbose");
  ghost = vm.count("ghost");
  return 1;
}
void show(const Mat& m) {
  imshow("show", m);
  waitKey(0);
}
void generate_points(const Mat& m, int sam_num, vector<Point>& _points) {
  const int rng = 3;
  _points.reserve(sam_num);
  while(_points.size() < sam_num) {
    const int r = rand()%(m.rows-10)+5;
    const int c = rand()%(m.cols-10)+5;
    const Vec3b& val = m.at<Vec3b>(r, c);
    bool same = true;
    for(int j = r-rng; j<=r+rng; ++j) for(int k = c-rng; k<=c+rng; ++k) {
      if(m.at<Vec3b>(j, k) != val) {
        same = false;
        break;
      }
    }
    if( same ) _points.emplace_back(c, r);
  }
}
void mycvtColor(const Mat& m, Mat& src) {
  vector<Mat> splits(3);
  split(m, splits);
  src = (splits[0]*0.114 + splits[1]*0.587 + splits[2]*0.299)/3;
}
double kernel(const Vec5d& v1, const Vec5d& v2) {
  double sum = 0;
  for(int i = 0; i<5; ++i) sum += (v1[i]-v2[i])*(v1[i]-v2[i]);
  double rtn = _2pi*exp(-0.5 * sum);
  return rtn;
}
void ghost_removal(const vector<Mat>& pics, vector<Mat>& result) {
  const int neigh = 1, iter = 15, pic_num = (int)pics.size();
  const int cols = pics[0].cols, rows = pics[0].rows;
  const Size sz = pics[0].size();
  Mat w(Size(256, 1), CV_64FC1, Scalar::all(0));
  for(int i = 0; i<256; ++i) w.at<double>(i) = (1 - pow(2.0*i/255-1, 12));

  vector<Mat> W(pic_num), merged(pic_num), W_Z(pic_num);
  vector< vector<Mat> > split_pics(pic_num);
  Mat x(sz, CV_64FC1, Scalar::all(0));
  Mat y(sz, CV_64FC1, Scalar::all(0));
  for(int i = 0; i<rows; ++i) for(int j = 0; j<cols; ++j) {
    x.at<double>(i, j) = double(j);//(cols-1);
    y.at<double>(i, j) = double(i);//(rows-1);
  }
  for(int i = 0; i<pic_num; ++i) {
    split_pics.push_back(vector<Mat>(3));
    cvtColor(pics[i], merged[i], COLOR_BGR2Lab);
    merged[i].convertTo(merged[i], CV_64FC3);
    //merged[i] /= 255.0;
    split(merged[i], split_pics[i]);
    split_pics[i].push_back(x);
    split_pics[i].push_back(y);
    merge(split_pics[i], merged[i]);

    vector<Mat> Ws(3);
    LUT(pics[i], w, W[i]);
    split(W[i], Ws);
    W[i] = (Ws[0]+Ws[1]+Ws[2])/3.0;
  }
  for(int i = 0; i<pic_num; ++i) W[i].copyTo(W_Z[i]);
  for(int i = 0; i<iter; ++i) {
    cerr << "iter: " << i << endl;
    vector<Mat> P;
    for(int j = 0; j<pic_num; ++j) {
      cerr << "pic: " << j << endl;
      P.push_back(Mat(sz, CV_64FC1, Scalar::all(0)));
      for(int c = neigh; c+neigh<cols; ++c) {
        for(int r = neigh; r+neigh<rows; ++r) {
          const Vec5d& v1 = merged[j].at<Vec5d>(r, c);
          double w_k = 0, sum_w = 0;
          for(int dc = c-neigh; dc<=c+neigh;++dc) {
            for(int dr = r-neigh; dr<=r+neigh; ++dr) if(dc != c || dr != r) {
              for(int p = 0; p<pic_num; ++p) {
                sum_w += W[p].at<double>(dr, dc);
                const Vec5d& v2 = merged[p].at<Vec5d>(dr, dc);
                w_k += W[p].at<double>(dr, dc) * kernel(v1, v2);
              }
            }
          }
          P[j].at<double>(r, c) = w_k / sum_w;
        }
      }
    }
    for(int j = 0; j<pic_num; ++j) W[j] = W_Z[j].mul(P[j]);
  }
  result.resize(pic_num);
  for(int i = 0; i<pic_num; ++i) W[i].copyTo(result[i]);
}
