#ifndef _BLENDING_HPP_
#define _BLENDING_HPP_

class BLENDING {
public:
  BLENDING(const vector<double>& para, const vector<Mat>& i, 
           vector<vector<Mat>>& s, const vector<vector<pair<int,int>>>& ord,
           vector<Mat>& ou)
    : _para(para), imgs(i), shift(s), order(ord), outputs(ou) {};
  void average();
  void linear();
  void multi_band();
private:
  void straightening();
  pair<Point2d, Point2d> get_corner(const Mat&, const Mat&);
  const vector<double>& _para;
  const vector<Mat>& imgs;
  const vector<vector<pair<int,int>>>& order;
  vector<vector<Mat>>& shift;
  vector<Mat>& outputs;
};
#endif
