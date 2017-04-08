#ifndef MTB_H
#define MTB_H

class MTB {

public:
  MTB(const vector<Mat>& pics) : _pics(pics) {};
  void process(vector<Mat>&, int, int);
private:
  void transform_bi(const Mat&, Mat&, Mat&, int);
  void shift(Mat&, Mat&, const pair<int, int>&);
  pair<int, int> align(const int, int, const int);

  const vector<Mat>& _pics;
  vector< vector<Mat> > _bi_pics, _masks;
};

#endif
