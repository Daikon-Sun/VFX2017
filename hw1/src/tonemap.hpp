#ifndef TONEMAP_H
#define TONEMAP_H

class TONEMAP {
public:
  TONEMAP(double f, double m, double a, double c)
    : _f(exp(-f)), _m(m), _a(a), _c(c) {
    //TODO: user parameter assertions
  };
  void process(Mat&, Mat&);
private:
  double _f;   // intensity
  double _m;   // contrast
  double _a;   // light adaption
  double _c;   // chromatic adaption
};

#endif
