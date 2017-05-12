#ifndef _MATCHING_HPP_
#define _MATCHING_HPP_

class MATCHING {
public:
  MATCHING(const int&, const vector<double>&, const vector<Mat>&, 
           vector<vector<Keypoint>>&, vector<vector<vector<pair<int,int>>>>&);
  void HAAR();
  void FLANN();
  void exhaustive();
private:
  void show_match();
  bool in_mid(const int&);
  bool is_align(const Keypoint&, const Keypoint&, const double&);
  bool check_match_haar(const tuple<int, int, double>&, 
                        size_t, size_t, double) const;
  bool check_match_exhaustive(int, int, size_t, size_t) const;
  const int& panorama_mode;
  vector<double> _para;
  const vector<Mat>& imgs;
  vector<vector<Keypoint>>& keypoints;
  vector<vector<vector<pair<int,int>>>>& match_pairs;
};

#endif
