#ifndef _TYPE_HPP_
#define _TYPE_HPP_

struct PreKeypoint {
  int     x;
  int     y;
  int     minR2;
  float   hm;
  PreKeypoint() {}
  PreKeypoint(int x, int y, float h) 
    : x(x), y(y), minR2(INT_MAX), hm(h) {}
};

struct Keypoint {
  float   x;
  float   y;
  float   l;
  float   t; // orientation angle in degreees
  Mat     patch;
  Keypoint() {}
  Keypoint(float x, float y, float l, float t) 
    : x(x), y(y), l(l), t(t) {}
};

#endif