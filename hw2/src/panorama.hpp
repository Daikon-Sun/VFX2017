#include "blending.hpp"
#include "detection.hpp"
#include "matching.hpp"
#include "projection.hpp"
#include "stitching.hpp"

#ifndef _PANORAMA_HPP_
#define _PANORAMA_HPP_

class PANORAMA : public DETECTION, public MATCHING, public PROJECTION, 
                 public STITCHING, public BLENDING {
public:
  PANORAMA(const string&, const string&, const double&,
           const int&, const int&, const int&, const int&, const int&, 
           const int&, const Para&, const Para&, const Para&, const Para&,
           const bool&);
  void process();
private:
  template<typename T> void execute(const T& f);
  void visualize();
  vector<Mat> _imgs;
  int _panorama_mode, _detection_mode, _matching_mode, _projection_mode;
  int _stitching_mode, _blending_mode;
  bool _verbose;
  string _out_prefix;
  vector<vector<Keypoint>> _keypoints;
  vector<vector<vector<pair<int,int>>>> _match_pairs;
  vector<vector<Mat>> _shift;
  vector<vector<pair<int,int>>> _order;
  vector<Mat> _outputs;
};

#endif
