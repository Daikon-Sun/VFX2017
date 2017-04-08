#ifndef DEBEVEC_H
#define DEBEVEC_H

class DEBEVEC {
public:
  DEBEVEC(const vector<Mat>& pics, const vector<double>& etimes)
    : _pics(pics), _etimes(etimes) {};
  void process(Mat&, double, vector<Mat>&);

private:
  const vector<Mat>& _pics;
  const vector<double>& _etimes;
};
class MERTENS {
public:
  MERTENS(const vector<Mat>& pics) : _pics(pics) {};
  void process(Mat&, double, double, double, vector<Mat>&);

private:
  const vector<Mat>& _pics;
};

#endif
