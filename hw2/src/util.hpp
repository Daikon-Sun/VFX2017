#ifndef _UTIL_HPP_
#define _UTIL_HPP_

struct PreKeypoint {
  int     x;
  int     y;
  int     minR2;
  double  hm;
  PreKeypoint() {}
  PreKeypoint(int x, int y, double h) 
    : x(x), y(y), minR2(INT_MAX), hm(h) {}
};

struct Keypoint {
  double   x;
  double   y;
  double   t; // orientation angle in degreees
  Mat     patch;
  Keypoint() {}
  Keypoint(double x, double y, double t) : x(x), y(y), t(t) {}
};

struct SIFTpoint {
  int x, y, l, oc;
  double t; // orientation angle in degreees
  SIFTpoint() {};
  SIFTpoint(int x, int y, int l, double t, int oc) 
    : x(x), y(y), l(l), t(t), oc(oc) {};

};

bool check(const vector<double>&, const string&,
           const int&, const string&);

int parse(int, char**);

#endif
