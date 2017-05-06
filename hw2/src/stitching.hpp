#ifndef _STITCHING_HPP_
#define _STITCHING_HPP_

class STITCHING {
public:
 STITCHING(const vector<double>& para, const vector<Mat>& i,
           const vector< vector<Keypoint> >& k,
           const vector<vector<pair<int, int> > >& m,
           vector< pair<double, double> >& s)
          : imgs(i), keypoints(k), match_pairs(m), shift(s), _para(para) {};
 void translation();
 void focal_length();
 void rotation();
private:
  bool is_inliner(size_t, double, double, const pair<int, int>&, double);
  bool is_inliner(size_t, const Mat&, const pair<int, int>&); 
  pair<double, double> cylindrical_projection(double, double, double, double, double);
  const vector<Mat>& imgs;
  const vector< vector<Keypoint> >& keypoints;
  const vector< vector<pair<int, int> > >& match_pairs;
  vector< pair<double, double> >& shift;
  vector<double> _para;
};

#endif
