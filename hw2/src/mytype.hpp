#ifndef _MYTYPE_HPP_
#define _MYTYPE_HPP_

#include <climits>

class Keypoint {
public:
  Keypoint() {}
  Keypoint(int x, int y, float hm)
    : _x(x), _y(y), _minR2(INT_MAX), _hm(hm) {}

  Point get_point() { return Point(_x, _y); }

  void update_minR2(Keypoint p, double r) {
    if (_hm > r * p._hm) return;
    int newR2 = pow(_x - p._x, 2) + pow(_y - p._y, 2);
    if (newR2 < _minR2) _minR2 = newR2; 
  }
  bool comp_minR2(Keypoint p) { return _minR2 > p._minR2; }

private:
  int               _x;
  int               _y;
  int               _minR2;
  float             _hm;
};

#endif