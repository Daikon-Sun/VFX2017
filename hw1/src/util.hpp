#ifndef UTIL_H
#define UTIL_H

bool check(string, size_t, int);
int parse(int, char**);
void generate_points(const Mat&, vector<Point>&);
void show(const Mat&);
void mycvtColor(const Mat&, Mat&);
void ghost_removal(const vector<Mat>&, vector<Mat>&);

#endif
