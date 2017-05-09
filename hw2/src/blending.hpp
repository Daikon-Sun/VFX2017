#ifndef _BLENDING_HPP_
#define _BLENDING_HPP_

class BLENDING {
public:
  BLENDING(const vector<Mat>& i, vector<vector<Mat>>& s, 
           const vector<vector<pair<int,int>>>& ord, vector<Mat>& ou)
    : imgs(i), shift(s), order(ord), outputs(ou) {};
  void linear();
private:
  pair<Point2d, Point2d> get_corner(Mat&, const Mat&);
  const vector<Mat>& imgs;
  const vector<vector<pair<int,int>>>& order;
  vector<vector<Mat>>& shift;
  vector<Mat>& outputs;
};
#endif
