#ifndef UTIL_H
#define UTIL_H

bool check(string, size_t, int);
int parse(int, char**);
void generate_points(const Mat&, int, vector<Point>&);
void show(const Mat&);
void mycvtColor(const Mat&, Mat&);
void ghost_removal(const vector<Mat>&, int, vector<Mat>&);
void blob_removal(const Mat&, Mat&);
void tune_blob(const Mat&);
void add_spotlight(vector<Mat>&, const vector<double>&);

#endif
