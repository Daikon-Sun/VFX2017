#ifndef _MATCHING_HPP_
#define _MATCHING_HPP_

constexpr double rt8 = 1/sqrt(8);
constexpr double rt2 = 1/sqrt(2);
const Mat haar = 
  (Mat_<float>(8, 8) << rt8, rt8, rt8, rt8, rt8, rt8, rt8, rt8,
                        rt8, rt8, rt8, rt8,-rt8,-rt8,-rt8,-rt8,
                        0.5, 0.5,-0.5,-0.5,   0,   0,   0,   0,
                          0,   0,   0,   0, 0.5, 0.5,-0.5,-0.5,
                        rt2,-rt2,   0,   0,   0,   0,   0,   0,
                          0,   0, rt2,-rt2,   0,   0,   0,   0,
                          0,   0,   0,   0, rt2,-rt2,   0,   0,
                          0,   0,   0,   0,   0,   0, rt2,-rt2);
const Mat haar_T = [&]() -> const Mat { Mat res; transpose(haar, res);
                                        return res.clone(); }();

class MATCHING {
public:
  MATCHING(const vector<double>&, vector< vector<Keypoint> >&,
           vector< vector< pair<int, int> > >&);
  void HAAR();
  void exhaustive();
private:
  bool in_mid(const int&);
  bool is_align(const Keypoint&, const Keypoint&);
  bool check_match_haar(const tuple<int, int, double>&, size_t, double) const;
  bool check_match_exhaustive(int, int, size_t);
  vector<double> _para;
  vector< vector<Keypoint> >& keypoints;
  vector< vector< pair<int, int> > >& match_pairs;
};

#endif
