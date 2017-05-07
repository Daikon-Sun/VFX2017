#ifndef _PROJECTION_HPP_
#define _PROJECTION_HPP_

class PROJECTION {
public:
  PROJECTION(const vector<double>& para, vector<Mat>& i,
             vector< vector<Keypoint> >& k) 
            : _para(para), keypoints(k), imgs(i) {};
  void cylindrical();
  void no_projection();
protected:
  void set_focal_length(double);
private:
  pair<double, double> projected_xy(double, double, double, double);
  vector<double> _para;
  vector<Mat>& imgs;
  vector< vector<Keypoint> >& keypoints;
};

#endif
