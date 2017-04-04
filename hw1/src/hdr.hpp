#include "opencv2/opencv.hpp"

#ifndef DEBEVEC_H
#define DEBEVEC_H

using namespace cv;
using namespace std;

class DEBEVEC {

public:
  DEBEVEC(const vector<Mat>& pics, const vector<double>& etimes,
          const vector<Point>& points) 
    : _pics(pics), _etimes(etimes), _points(points) {};
  void process(Mat&, double);

private:
  int W(int val) { return (val<=127 ? val+1 : 256-val); }
  const vector<Mat>& _pics;
  const vector<float>& _etimes;
  const vector<Point>& _points;
};
#endif
