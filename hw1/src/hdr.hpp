#ifndef DEBEVEC_H
#define DEBEVEC_H

class DEBEVEC {
public:
  DEBEVEC(const vector<double>& para) : _para(para) {};
  void process(const vector<Mat>&, const vector<double>&,
               const vector<Mat>&, Mat&);

private:
  const vector<double> _para;
};
class MERTENS {
public:
  MERTENS(const vector<double>& para) : _para(para) {};
  void process(const vector<Mat>&, const vector<Mat>&, Mat&);

private:
  const vector<double> _para;
};

#endif
