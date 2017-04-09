#include "opencv2/opencv.hpp"

#ifndef UTIL_H
#define UTIL_H

using namespace std;
using namespace cv;

void generate_points(const Mat&, vector<Point>&);
void show(const Mat&);
void mycvtColor(const Mat&, Mat&);
void ghost_removal(const vector<Mat>&, vector<Mat>&);

#endif
