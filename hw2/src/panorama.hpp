#include "detection.hpp"
#include "matching.hpp"
#include "projection.hpp"
#include "stitching.hpp"

#ifndef _PANORAMA_HPP_
#define _PANORAMA_HPP_

class PANORAMA : public DETECTION, public MATCHING, public PROJECTION, public STITCHING {
public:
  PANORAMA(const string&, const string&, const int&, const int&, const int&,
           const int&, const Para&, const Para&, const Para&);
  void process();
  void visualize();
private:
  template<typename T> void execute(const T& f);
  vector<Mat> _imgs;
  int _detection_method, _matching_method, _projection_method, _stitching_method;
  string _out_jpg;
  vector< vector< Keypoint> > _keypoints;
  vector< vector< pair<int, int> > > _match_pairs;
  vector< pair<float, float> > _shift;
};

#endif
