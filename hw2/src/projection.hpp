#ifndef _PROJECTION_HPP_
#define _PROJECTION_HPP_

class PROJECTION {
public:
  PROJECTION(const vector<float>& para, vector<Mat>& i,
             vector< vector<Keypoint> >& k) 
            : _para(para), keypoints(k), imgs(i) {};
  void cylindrical();
  void no_projection();
private:
  pair<float, float> projected_xy(float, float, float, float);
  vector<float> _para;
  vector<Mat>& imgs;
  vector< vector<Keypoint> >& keypoints;
};

#endif
