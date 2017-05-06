#ifndef _STITCHING_HPP_
#define _STITCHING_HPP_

class STITCHING {
public:
 STITCHING(const vector<float>& para, const vector<Mat>& i,
           const vector< vector<Keypoint> >& k,
           const vector<vector<pair<int, int> > >& m,
           vector< pair<float, float> >& s)
          : imgs(i), keypoints(k), match_pairs(m), shift(s), _para(para) {};
 void translation();
 void focal_length();
 void rotation();
private:
  bool is_inliner(size_t, float, float, const pair<int, int>&, float);
  bool is_inliner(size_t, const Mat&, const pair<int, int>&); 
  pair<float, float> cylindrical_projection(float, float, float, float, float);
  const vector<Mat>& imgs;
  const vector< vector<Keypoint> >& keypoints;
  const vector< vector<pair<int, int> > >& match_pairs;
  vector< pair<float, float> >& shift;
  vector<float> _para;
};

#endif
