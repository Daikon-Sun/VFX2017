#include "detection.hpp"
#include "matching.hpp"
#include "projection.hpp"
#include "stitching.hpp"

#ifndef _PANORAMA_HPP_
#define _PANORAMA_HPP_

class PANORAMA : public DETECTION, public MATCHING, public PROJECTION, public STITCHING {
public:
  PANORAMA(const string&, const string&, 
           const int&, const int&, const int&, const int&, const int&, 
           const Para&, const Para&, const Para&, const bool&);
  void process();
  void visualize();
private:
  template<typename T> void execute(const T& f);
  vector<Mat> _imgs;
  int _panorama_mode;
  int _detection_mode, _matching_mode, _projection_mode, _stitching_mode;
  bool _verbose;
  string _out_jpg;
  vector<vector<Keypoint>> _keypoints;
  vector<vector<vector<pair<int,int>>>> _match_pairs;
  vector<vector<Mat>> _shift;
};

#endif
