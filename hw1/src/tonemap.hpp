#ifndef TONEMAP_H
#define TONEMAP_H

class TONEMAP {
public:
  TONEMAP(const vector<double>& para)
    : _f(exp(-para[0])), _m(para[1]), _a(para[2]), _c(para[3]) {};
  void process(Mat&, Mat&);
private:
  double _f;   // intensity
  double _m;   // contrast
  double _a;   // light adaption
  double _c;   // chromatic adaption
};

#endif
