#ifndef _MATCHING_HPP_
#define _MATCHING_HPP_

class MATCHING {
public:
  MATCHING(const vector<double>&, const vector<Mat>&, 
           vector< vector<Keypoint> >&, vector< vector< pair<int, int> > >&);
  void HAAR();
  void exhaustive();
private:
  bool in_mid(const int&);
  bool is_align(const Keypoint&, const Keypoint&);
  bool check_match_haar(const tuple<int, int, double>&, size_t, double) const;
  bool check_match_exhaustive(int, int, size_t);
  vector<double> _para;
  const vector<Mat>& imgs;
  vector< vector<Keypoint> >& keypoints;
  vector< vector< pair<int, int> > >& match_pairs;
};

#endif
