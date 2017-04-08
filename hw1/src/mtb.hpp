#ifndef MTB_H
#define MTB_H

class MTB {

public:
  MTB(const vector<int>& para) : _para(para) {};
  void process(const vector<Mat>&, vector<Mat>&);
private:
  void transform_bi(const Mat&, Mat&, Mat&, int);
  void shift(Mat&, Mat&, const pair<int, int>&);
  pair<int, int> align(const int, int, const int);

  const vector<int>& _para;
  vector< vector<Mat> > _bi_pics, _masks;
};

#endif
