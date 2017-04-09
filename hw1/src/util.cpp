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
extern int method, hdr_type, tonemap_type, fusion_type, ghost;
extern bool verbose, blob, blob_tune;
extern vector<int> algn;
extern vector<double> spotlight, hdr_para, tonemap_para, fusion_para;

inline bool check(string str, size_t sz, int i) {
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
      ("blob,b", value<bool>()->implicit_value(blob, blob?"True":"False"),
       "Add blob-removal.")
      ("blob_tune", value<bool>()
       ->implicit_value(blob_tune, blob_tune?"True":"False")->composing(),
       "Tune blob-removal parameters.")
      ("ghost,g", value<int>(&ghost)->default_value(ghost), 
       "The number of iterations for ghost-removal.")
      ("spotlight,s", value< vector<double> >(&spotlight)->multitoken(),
       "The regions of interest of the first image to be enhanced.")
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
  if(!algn.empty() && algn[0] >= 0) {
    if(!check("Aligment", algn.size(), 2)) return -1;
  }
  if(!spotlight.empty()) {
    if(!check("Spotlight", spotlight.size()%4, 0)) return -1;
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
  blob = vm.count("blob");
  blob_tune = vm.count("blob_tune");
  return 1;
}
void generate_points(const Mat& m, int sam_num, vector<Point>& _points) {
  while(_points.size() < sam_num) {
    const int r = rand()%(m.rows-10)+5;
    const int c = rand()%(m.cols-10)+5;
    _points.emplace_back(c, r);
  }
}
void mycvtColor(const Mat& m, Mat& src) {
  vector<Mat> splits(3);
  split(m, splits);
  src = (splits[0]*0.114 + splits[1]*0.587 + splits[2]*0.299)/3;
}
inline double kernel(const Vec5d& v1, const Vec5d& v2) {
  double sum = 0;
  for(int i = 0; i<5; ++i) sum += (v1[i]-v2[i])*(v1[i]-v2[i]);
  return _2pi*exp(-0.5 * sum);
}
void
ghost_removal(const vector<Mat>& pics, const int iter, vector<Mat>& result) {
  const int neigh = 1, pic_num = (int)pics.size();
  const int cols = pics[0].cols, rows = pics[0].rows;
  const Size sz = pics[0].size();
  Mat w(1, 256, CV_64FC1);
  for(int i = 0; i<256; ++i) w.at<double>(i) = (1 - pow(2.0*i/255-1, 12));

  vector<Mat> W(pic_num), merged(pic_num), W_Z(pic_num);
  vector< vector<Mat> > split_pics(pic_num);
  Mat x(sz, CV_64FC1, Scalar::all(0));
  Mat y(sz, CV_64FC1, Scalar::all(0));
  for(int i = 0; i<rows; ++i) for(int j = 0; j<cols; ++j) {
    x.at<double>(i, j) = double(j)/(cols-1);
    y.at<double>(i, j) = double(i)/(rows-1);
  }
  for(int i = 0; i<pic_num; ++i) {
    split_pics.push_back(vector<Mat>(3));
    cvtColor(pics[i], merged[i], COLOR_BGR2Lab);
    merged[i].convertTo(merged[i], CV_64FC3);
    merged[i] /= 255.0;
    split(merged[i], split_pics[i]);
    split_pics[i].push_back(x.clone());
    split_pics[i].push_back(y.clone());
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
      //cerr << "pic: " << j << endl;
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
    for(int j = 0; j<pic_num; ++j) {
      W[j] = W_Z[j].mul(P[j]);
    }
  }
  result.resize(pic_num);
  for(int i = 0; i<pic_num; ++i) W[i].copyTo(result[i]);
}
void blob_removal(const Mat& pic, Mat& result) {
  
  const int cols = pic.cols, rows = pic.rows;
  Mat res;
  pic.copyTo(res);
  int lowL = 0, lowA = 0, lowB = 0, highL = 40, highA = 255, highB = 255;
  for(; highL <= 72; highL+=4) {
    Mat m;
    cvtColor(res, m, CV_BGR2Lab);
    SimpleBlobDetector::Params params;
    inRange(m, Scalar(lowL, lowA, lowB), Scalar(highL, highA, highB), m);
    
    Mat se = getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7), Point(-1, -1));
    morphologyEx(m, m, MORPH_CLOSE, se);
    bitwise_not(m, m);

    int minA = 400, maxA = 2500;
    double minCircularity = 0.41, minConvexity = 0.41, minInertiaRatio = 0.41;
    params.filterByArea = true;
    params.minArea = minA;
    params.maxArea = maxA;
    params.filterByCircularity = true;
    params.minCircularity = minCircularity;
    params.filterByConvexity = true;
    params.minConvexity = minConvexity;
    params.filterByInertia = true;
    params.minInertiaRatio = minInertiaRatio;
    params.minDistBetweenBlobs = 500;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;
    detector->detect(m, keypoints);
    const double f = 0.9;
    for(auto& k:keypoints) {
      if(k.pt.x<f*1.5*k.size || k.pt.y<f*1.5*k.size ||
         k.pt.x+f*1.5*k.size>=cols || k.pt.y+f*1.5*k.size>=rows) continue;
      Mat target(Size(f*k.size, f*k.size), CV_64FC3, Scalar::all(0));
      for(int c = -1; c<=1; ++c) for(int r = -1; r<=1; ++r) {
        if(c || r) {
          int cc = k.pt.x-f/2*k.size+c*f*k.size;
          int rr = k.pt.y-f/2*k.size+r*f*k.size;
          Rect roi(cc, rr, f*k.size, f*k.size);
          Mat tmp = res(roi).clone();
          tmp.convertTo(tmp, CV_64FC3);
          target += tmp;
        }
      }
      target /= 8;
      target.convertTo(target, CV_8UC3);
      target.copyTo(
          res(Rect(k.pt.x-f/2*k.size, k.pt.y-f/2*k.size, f*k.size, f*k.size)));
    }
  }
  res.copyTo(result);
}
void tune_blob(const Mat& img1) {
  
  Mat img2, img3, display;
  cvtColor(img1, img2, CV_BGR2Lab);

  namedWindow("thresh", WINDOW_NORMAL);
  namedWindow("blob", WINDOW_NORMAL);

  int lowL = 150, lowA = 0, lowB = 155, highL = 255, highA = 255, highB = 255;
  int min_dis = 100;
  createTrackbar("lowL", "thresh", &lowL, 255);
  createTrackbar("lowA", "thresh", &lowA, 255);
  createTrackbar("lowB", "thresh", &lowB, 255);
  createTrackbar("highL", "thresh", &highL, 255);
  createTrackbar("highA", "thresh", &highA, 255);
  createTrackbar("highB", "thresh", &highB, 255);

  int minArea = 35, maxArea = 172, minCircularity = 58;
	int minConvexity = 87, minInertiaRatio = 21;

  createTrackbar("minArea", "blob", &minArea, 1000);
  createTrackbar("maxArea", "blob", &maxArea, 4000);
  createTrackbar("minCircular", "blob", &minCircularity, 99);
  createTrackbar("minConvex", "blob", &minConvexity, 99);
  createTrackbar("minInertia", "blob", &minInertiaRatio, 99);
  createTrackbar("min_dis", "blob", &min_dis, 1000);

  SimpleBlobDetector::Params params;
  vector<KeyPoint> keypoints;

  while (waitKey(1) != 27) //press 'esc' to quit
  {
      inRange(img2, Scalar(lowL, lowA, lowB),
              Scalar(highL, highA, highB), img2);

      Mat se = getStructuringElement(CV_SHAPE_ELLIPSE, Size(7, 7), Point(3, 3));
      morphologyEx(img2, img2, MORPH_CLOSE, se);
      imshow("thresh", img2);

      bitwise_not(img2, img3);

      params.filterByArea = true;
      params.minArea = minArea + 1;
      params.maxArea = maxArea + 1;

      params.filterByCircularity = true;
      params.minCircularity = (minCircularity + 1) / 100.0;

      params.filterByConvexity = true;
      params.minConvexity = (minConvexity + 1) / 100.0;

      params.filterByInertia = true;
      params.minInertiaRatio = (minInertiaRatio + 1) / 100.0;

      params.minDistBetweenBlobs = min_dis;

			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
      detector->detect(img3, keypoints);
      drawKeypoints(img1, keypoints, display,
                    Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

      stringstream displayText;
      displayText = stringstream();
      displayText << "Blob_count: " << keypoints.size();
      putText(display, displayText.str(), Point(0, 50),
              CV_FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
      imshow("blob", display);
  }
}
void add_spotlight(vector<Mat>& pics, const vector<double>& para) {
  int pic_num = (int)pics.size();
  for(int i = 1; i<pic_num; ++i) {
    for(int j = 0; j<(int)para.size(); j+=4) {
      Mat res;
      Rect roi(para[j], para[j+1], para[j+2], para[j+3]);
      cerr << roi << endl;
      Mat msk(pics[0].size(), CV_8UC1, Scalar::all(0));
      msk(roi) = 255;
      seamlessClone(pics[0], pics[i], msk,
                    Point(para[j]+para[j+2]/2, para[j+1]+para[j+3]/2),
                    pics[i], 1);
    }
  }
}
