#ifndef _STITCHING_HPP_
#define _STITCHING_HPP_

class STITCHING {
public:
 STITCHING(const int& p_mode, const vector<double>& para, const vector<Mat>& i,
           const vector<vector<Keypoint>>& k,
           const vector<vector<vector<pair<int,int>>>>& m,
           vector<vector<Mat>>& s)
          : panorama_mode(p_mode), imgs(i), keypoints(k), match_pairs(m),
            shift(s), _para(para) {};
 void translation();
 void focal_length();
 void rotation();
 void autostitch();
private:
  bool is_inliner(size_t, size_t, double, double, const pair<int, int>&, double);
  bool is_inliner(size_t, size_t, const Mat&, const pair<int, int>&); 
  pair<double, double> cylindrical_projection(double, double, double, double, double);
  const int& panorama_mode;
  const vector<Mat>& imgs;
  const vector<vector<Keypoint>>& keypoints;
  const vector<vector<vector<pair<int,int>>>>& match_pairs;
  vector<vector<Mat>>& shift;
  vector<double> _para;
};

#endif
