#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#include "util.hpp"
#include "stitching.hpp"

bool STITCHING::is_inliner(size_t p1, size_t p2, double sx, double sy,
                           const pair<int, int>& kpid, double f = 0) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[p1][kpid1], kp2 = keypoints[p2][kpid2];
  double _sx, _sy;
  if(f > 0) {
    Size sz1 = imgs[p1].size(), sz2 = imgs[p2].size();
    double x1, y1;
    tie(x1, y1) = cylindrical_projection(f, sz1.width, sz1.height, kp1.x, kp1.y);
    double x2, y2;
    tie(x2, y2) = cylindrical_projection(f, sz2.width, sz2.height, kp2.x, kp2.y);
    _sx = x1 - x2, _sy = y1 - y2;
  } else _sx = kp1.x-kp2.x, _sy = kp1.y-kp2.y;
  return (sx-_sx) * (sx-_sx) + (sy-_sy) * (sy-_sy) < _para[1];
}
bool STITCHING::is_inliner(size_t p1, size_t p2, const Mat& sol,
                           const pair<int, int>& kpid) {
  int kpid1, kpid2; tie(kpid1, kpid2) = kpid;
  const auto& kp1 = keypoints[p1][kpid1], kp2 = keypoints[p2][kpid2];
  Mat pos2 = (Mat_<double>(3, 1) << kp2.x, kp2.y, 1);
  pos2 = sol * pos2;
  Mat pos1 = (Mat_<double>(2, 1) << kp1.x, kp1.y);
  Mat err = pos1 - pos2;
  return sum(err.mul(err))[0] < _para[1];
}
pair<double, double>
STITCHING::cylindrical_projection(double f, double w, double h, 
                                  double x, double y) {
  return {f*atan((x-w/2)/f) + w/2, f*(y-h/2)/sqrt(f*f+(x-w/2)*(x-w/2)) + h/2};
}
void STITCHING::translation() { 
  cerr << __func__;
  gen_order();
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  #pragma omp parallel for
  for(size_t p1 = 0; p1<pic_num; ++p1) {
    shift[p1][p1] = Mat::eye(3, 3, CV_64FC1);
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      int best_cnt = 0;
      double best_sx, best_sy;
      const size_t sz = match_pairs[p1][p2].size();
      assert(sz);
      for(int i = 0; i<int(_para[0]); ++i) {
        size_t id1 = rand()%sz;
        int kpid1, kpid2; tie(kpid1, kpid2) = match_pairs[p1][p2][id1];
        const auto& kp1 = keypoints[p1][kpid1], kp2 = keypoints[p2][kpid2];
        double sx = kp1.x-kp2.x, sy = kp1.y-kp2.y;
        int in_cnt = 0;
        for(size_t id3 = 0; id3<match_pairs[p1][p2].size(); ++id3)
          if(id3 != id1) 
            in_cnt += is_inliner(p1, p2, sx, sy, match_pairs[p1][p2][id3]);
        if(in_cnt > best_cnt) {
          best_cnt = in_cnt;
          best_sx = sx;
          best_sy = sy;
        }
      }
      shift[p1][p2] = (Mat_<double>(3, 3) << 1, 0, best_sx,
                                             0, 1, best_sy,
                                             0, 0,       1);
      shift[p2][p1] = shift[p1][p2].inv();
      if(!panorama_mode) break;
    }
  }
}
void STITCHING::focal_length() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  #pragma omp parallel for
  for(size_t p1 = 0; p1<pic_num; ++p1)
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      int best_cnt1 = 0, best_cnt2 = 0;
      const size_t sz = match_pairs[p1][p2].size();
      Size sz1 = imgs[p1].size(), sz2 = imgs[p2].size();
      double best_f = 0, best_sx, best_sy;
      double w1 = sz1.width/2.0, w2 = sz2.width/2.0;
      double h1 = sz1.height/2.0, h2 = sz2.height/2.0;
      for(int i = 0; i<int(_para[0]); ++i) {
        size_t id1 = rand()%sz;
        size_t id2 = (id1+(rand()%(sz-1))+1)%sz;
        int kpid11, kpid21; tie(kpid11, kpid21) = match_pairs[p1][p2][id1];
        int kpid12, kpid22; tie(kpid12, kpid22) = match_pairs[p1][p2][id2];
        const auto& kp11 = keypoints[p1][kpid11];
        const auto& kp21 = keypoints[p2][kpid21];
        const auto& kp12 = keypoints[p1][kpid12];
        const auto& kp22 = keypoints[p2][kpid22];
        double nu = (kp21.x-w2) * (kp22.x-w2) * (kp12.x-kp11.x)
                  -(kp11.x-w1) * (kp12.x-w1) * (kp22.x-kp21.x);
        double de = (kp22.x-kp21.x + kp11.x-kp12.x);
        if(abs(de) >= 25 || nu*de < 0) continue;
        double f = sqrt(nu / de);
        double nx11, ny11; 
        tie(nx11, ny11) = cylindrical_projection(f, w1*2, h1*2, kp11.x, kp11.y);
        double nx12, ny12; 
        tie(nx12, ny12) = cylindrical_projection(f, w1*2, h1*2, kp12.x, kp12.y);
        double nx21, ny21; 
        tie(nx21, ny21) = cylindrical_projection(f, w2*2, h2*2, kp21.x, kp21.y);
        double nx22, ny22;
        tie(nx22, ny22) = cylindrical_projection(f, w2*2, h2*2, kp22.x, kp22.y);
        double sx1 = (nx11-nx21), sx2 = (nx12-nx22);
        double sy1 = (ny11-ny21), sy2 = (ny12-ny22);
        if(abs(sx1-sx2) > 5 || abs(sy1-sy2) > 5) continue;
        double sx = (sx1+sx2)/2, sy = (sy1+sy2)/2;
        int in_cnt1 = 0, in_cnt2 = 0;
        for(size_t id3 = 0; id3<match_pairs[p1][p2].size(); ++id3)
          if(id3 != id1) {
            in_cnt1 += is_inliner(p1, p2, sx, sy, match_pairs[p1][p2][id3]);
            _para[0] /= 2;
            in_cnt2 += is_inliner(p1, p2, sx, sy, match_pairs[p1][p2][id3]);
            _para[0] *= 2;
          }
        if(in_cnt1>best_cnt1 || (in_cnt1==best_cnt1 && in_cnt2<best_cnt2)) {
          best_cnt1 = in_cnt1;
          best_cnt2 = in_cnt2;
          best_sx = sx;
          best_sy = sy;
          best_f = f;
        }
      }
      shift[p1][p2] = (Mat_<double>(3, 3) << 1, best_f, best_sx,
                                             0,      1, best_sy,
                                             0,      0,       1);
      if(!panorama_mode) break;
    }
}
void STITCHING::rotation() {
  cerr << __func__;
  size_t pic_num = imgs.size();
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  #pragma omp parallel for
  for(size_t p1 = 0; p1<pic_num; ++p1)
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      Mat best_sol;
      int best_cnt1 = 0, best_cnt2 = 0;
      const size_t sz = match_pairs[p1][p2].size();
      for(int i = 0; i<int(_para[0]); ++i) {
        size_t id1 = rand()%sz;
        size_t id2 = (id1+(rand()%(sz-1))+1)%sz;
        size_t id3 = rand()%sz;
        while(id3 == id1 || id3 == id2) id3 = rand()%sz;
        int kpid11, kpid12; tie(kpid11, kpid12) = match_pairs[p1][p2][id1];
        int kpid21, kpid22; tie(kpid21, kpid22) = match_pairs[p1][p2][id2];
        int kpid31, kpid32; tie(kpid31, kpid32) = match_pairs[p1][p2][id3];
        const auto& kp11 = keypoints[p1][kpid11];
        const auto& kp21 = keypoints[p1][kpid21];
        const auto& kp31 = keypoints[p1][kpid31];
        const auto& kp12 = keypoints[p2][kpid12];
        const auto& kp22 = keypoints[p2][kpid22];
        const auto& kp32 = keypoints[p2][kpid32];
        Mat rot = (Mat_<double>(3, 3) << kp12.x, kp22.x, kp32.x,
                                         kp12.y, kp22.y, kp32.y,
                                            1.0,    1.0,    1.0);
        Mat pos = (Mat_<double>(2, 3) << kp11.x, kp21.x, kp31.x,
                                         kp11.y, kp21.y, kp31.y);
        Mat sol = pos * rot.inv();
        int in_cnt1 = 0, in_cnt2 = 0;
        for(size_t id3 = 0; id3<match_pairs[p1][p2].size(); ++id3)
          if(id3 != id1) {
            in_cnt1 += is_inliner(p1, p2, sol, match_pairs[p1][p2][id3]);
            _para[1] /= 2;
            in_cnt2 += is_inliner(p1, p2, sol, match_pairs[p1][p2][id3]);
            _para[1] *= 2;
          }
        if(in_cnt1>best_cnt1 || (in_cnt1==best_cnt1 && in_cnt2>best_cnt2)) {
          best_cnt1 = in_cnt1;
          best_cnt2 = in_cnt2;
          sol.copyTo(best_sol);
        }
      }
      copyMakeBorder(best_sol, best_sol, 0, 1, 0, 0, BORDER_CONSTANT, 0);
      best_sol.at<double>(2, 2) = 1;
      shift[p1][p2] = best_sol.clone();
      if(!panorama_mode) break;
    }
}
void STITCHING::homography() {
  cerr << __func__ << endl;
  gen_order();
  size_t pic_num = imgs.size();
  vector<vector<pair<int, int>>> in_cnt(pic_num, vector<pair<int,int>>(pic_num));
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  #pragma omp parallel for 
  for(size_t p1 = 0; p1<pic_num; ++p1) {
    shift[p1][p1] = Mat::eye(3, 3, CV_64FC1);
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      if(match_pairs[p1][p2].size() < 5) continue;
      vector<Point2f> src, dst;      
      for(const auto& mp : match_pairs[p1][p2]) {
        int k1, k2; tie(k1, k2) = mp;
        const auto& kp1 = keypoints[p1][k1];
        const auto& kp2 = keypoints[p2][k2];
        src.emplace_back(kp2.x, kp2.y);
        dst.emplace_back(kp1.x, kp1.y);
      }
      Mat msk;
      shift[p1][p2] = findHomography(src, dst, CV_RANSAC, _para[0], msk);
      if(shift[p1][p2].empty()) continue;
      shift[p2][p1] = shift[p1][p2].inv();
      if(!panorama_mode) break;
    }
  }
}
struct BA {
  BA(double p1_x, double p1_y, double p2_x, double p2_y) 
    : p1_x(p1_x), p1_y(p1_y), p2_x(p2_x), p2_y(p2_y) {}

  template <typename T>
  bool operator()(const T* const R1, const T* const K1, const T* const R2,
                  const T* const K2, T* residuals) const {
    Eigen::Matrix<T, 3, 1> pos, pred;
    pos    << T(p2_x), T(p2_y), T(1);

    Eigen::Matrix<T, 3, 3> Ri, Rj, Ki, Kj_inv;// RI, RJ, Ris, Rjs;

    T ni = sqrt(R1[0]*R1[0] + R1[1]*R1[1] + R1[2]*R1[2]);
    T nj = sqrt(R2[0]*R2[0] + R2[1]*R2[1] + R2[2]*R2[2]);

    Ri     <<     T(0), T(-R1[2]),  T(R1[1]),
              T(R1[2]),      T(0), T(-R1[0]),
             T(-R1[1]),  T(R1[0]),      T(0);
    Ri /= ni;
    Ri = Eigen::Matrix<T, 3, 3>::Identity() 
       + sin(ni) * Ri + (T(1)-cos(ni)) * Ri * Ri;

    Rj   <<      T(0), T(-R2[2]), T(R2[1]),
             T(R2[2]),      T(0),T(-R2[0]),
            T(-R2[1]),  T(R2[0]),     T(0);
    Rj /= nj;
    Rj = Eigen::Matrix<T, 3, 3>::Identity()
       + sin(nj) * Rj + (T(1)-cos(nj)) * Rj * Rj;

    Ki     << K1[0],  T(0), T(0),
               T(0), K1[0], T(0),
               T(0),  T(0), T(1);

    Kj_inv << T(1.0)/K2[0],         T(0), T(0),
                      T(0), T(1.0)/K2[0], T(0),
                      T(0),         T(0), T(1);
    pred = Ki * Ri * Rj.transpose() * Kj_inv * pos;
    pred(0, 0) /= pred(2, 0);
    pred(1, 0) /= pred(2, 0);
    residuals[0] = T(sqrt((pred(0, 0) - p1_x)*(pred(0, 0) - p1_x) +
                          (pred(1, 0) - p1_y)*(pred(1, 0) - p1_y)));
    return true;
  }

   static ceres::CostFunction* Create(const double p1_x, const double p1_y,
                                      const double p2_x, const double p2_y) {
     return (new ceres::AutoDiffCostFunction<BA, 1, 3, 1, 3, 1>(
                 new BA(p1_x, p1_y, p2_x, p2_y)));
   }
  double p1_x, p1_y, p2_x, p2_y;
};
constexpr int M = 6;
void STITCHING::gen_order() {
  size_t pic_num = imgs.size();
  vector<vector<pair<int, int>>> in_cnt(pic_num, vector<pair<int,int>>(pic_num));
  shift.clear();
  shift.resize(pic_num, vector<Mat>(pic_num));
  inners.resize(pic_num, vector<vector<pair<int,int>>>(pic_num));
  #pragma omp parallel for 
  for(size_t p1 = 0; p1<pic_num; ++p1) {
    for(size_t p2 = p1+1; p2<pic_num; ++p2) {
      if(match_pairs[p1][p2].empty()) continue;
      vector<Point2f> src, dst;      
      for(const auto& mp : match_pairs[p1][p2]) {
        int k1, k2; tie(k1, k2) = mp;
        const auto& kp1 = keypoints[p1][k1];
        const auto& kp2 = keypoints[p2][k2];
        src.emplace_back(kp2.x, kp2.y);
        dst.emplace_back(kp1.x, kp1.y);
      }
      Mat msk;
      findHomography(src, dst, CV_RANSAC, _para[0], msk);
      int sm = sum(msk)[0];
      in_cnt[p1][p2] = {sm, p2};
      in_cnt[p2][p1] = {sm, p1};
      for(size_t j = 0; j<msk.rows; ++j) if(msk.at<uchar>(j, 0))
        inners[p1][p2].push_back(match_pairs[p1][p2][j]);
    }
  }
  edges.resize(pic_num);
  vector<pair<int,int>> cnt_all(pic_num);
  for(size_t i = 0; i<pic_num; ++i) {
    cnt_all[i].second = i;
    sort(in_cnt[i].begin(), in_cnt[i].end(), greater<pair<int,int>>()); 
    for(size_t j = 0; j<M && j<in_cnt[i].size(); ++j) {
      int p1 = i, p2 = in_cnt[i][j].second;
      if(p1 > p2) swap(p1, p2);
      if(in_cnt[i][j].first > 5.9+0.22*match_pairs[p1][p2].size()) {
        edges[p1].insert(p2);
        edges[p2].insert(p1);
        cnt_all[p1].first += in_cnt[i][j].first;
        cnt_all[p2].first += in_cnt[i][j].first;
      }
    }
  }
  vector<bool> vis(pic_num);
  size_t finish_cnt = 0;
  while(finish_cnt < pic_num) {
    vector<pair<int,int>> ord;
    int root = max_element(cnt_all.begin(), cnt_all.end())->second;
    vis[root] = true;
    priority_queue<tuple<int,int,int>> pq;
    pq.emplace(0, root, -1);
    while(!pq.empty()) {
      int u, p; tie(ignore, u, p) = pq.top(); pq.pop();
      ord.emplace_back(u, p);
      cnt_all[u].first = -1;
      ++finish_cnt;
      for(auto v : edges[u]) if(!vis[v]) {
        vis[v] = true;
        pq.emplace(in_cnt[u][v].first, v, u);
      } 
    }
    order.push_back(ord);
  }
}
void STITCHING::autostitch() {
  cerr << __func__;
  gen_order();
  //for(auto& y : order) for(auto x:y)
  //  cerr << x.first << " " << x.second << endl;

  size_t pic_num = imgs.size();
  using ceres::AutoDiffCostFunction;
  using ceres::CostFunction;
  using ceres::Problem;
  using ceres::Solver;
  using ceres::Solve;
  using ceres::LossFunction;
    
  double R[pic_num][3] = {0};
  for(size_t i = 0; i<pic_num; ++i) fill_n(R[i], 3, 1e-10);
  double K[pic_num][1] = {0};
  for(const auto& ord : order) {
    size_t en = 1;
    K[ord[0].first][0] = _para[1];
    while(en < ord.size()) {
      vector<int> group;
      for(size_t st = 0; st < en; ++st) group.push_back(ord[st].first);
      for(size_t j = 0; j<3; ++j) R[ord[en].first][j] = R[ord[en].second][j];
      K[ord[en].first][0] = K[ord[en].second][0];
      group.push_back(ord[en].first);
      ++en;

      Problem problem;
      ceres::LossFunctionWrapper* loss_func =
        new ceres::LossFunctionWrapper(new ceres::HuberLoss(100001),
                                       ceres::TAKE_OWNERSHIP);
      for(size_t i = 0; i<group.size(); ++i) {
        const int& p1 = group[i];
        for(auto p2 : edges[p1]) if(p1 != p2) {
          if(find(group.begin(), group.end(), p2) == group.end()) continue;
          bool rev = (p1 > p2);
          const auto& keypairs = (rev ? inners[p2][p1] : inners[p1][p2]);
          for(auto& keypair : keypairs) {
            int k1, k2; tie(k1, k2) = keypair;
            if(rev) swap(k1, k2);
            const auto& kp1 = keypoints[p1][k1];
            const auto& kp2 = keypoints[p2][k2];
            CostFunction* cost_func = BA::Create(kp1.x, kp1.y, kp2.x, kp2.y);
            problem.AddResidualBlock(cost_func, loss_func,
                                     R[p1], K[p1], R[p2], K[p2]);
          }
        }
      }
      const int tot_iter = (ord.size()-en)+8, max_iter = 500;
      Solver::Options options;
      options.max_num_iterations = max_iter;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.minimizer_progress_to_stdout = false;
      Solver::Summary summary;
      for(size_t iter = 0; iter < tot_iter; ++iter) {
        Solve(options, &problem, &summary);
        loss_func->Reset(
            new ceres::HuberLoss(100001/(iter*(10000.0/tot_iter)+1)+1),
            ceres::TAKE_OWNERSHIP
        );
      }
      //cerr << summary.FullReport() << endl;
    }
    //for(size_t i = 0; i<pic_num; ++i) {
    //  cerr << i << endl;
    //  for(size_t j = 0; j<3; ++j) cerr << R[i][j] << " ";
    //  cerr << K[i][0] << endl;
    //  cerr << "#####################" << endl;
    //}
  }
  vector<Mat> Rs(pic_num), Ks(pic_num), R_Ts(pic_num);
  #pragma omp parallel for
  for(size_t i = 0; i<pic_num; ++i) {
    Mat r = (Mat_<double>(3, 3) << 0, -R[i][2], R[i][1],
                                    R[i][2], 0, -R[i][0],
                                    -R[i][1], R[i][0], 0);
    double norm = sqrt(R[i][0]*R[i][0] + R[i][1]*R[i][1] + R[i][2]*R[i][2]);
    r = r / norm;
    Rs[i] = Mat::eye(3, 3, CV_64FC1) + sin(norm) * r + (1.0-cos(norm)) * r * r;
    transpose(Rs[i], R_Ts[i]);
    Ks[i] = (Mat_<double>(3, 3) << K[i][0], 0, 0,
                                     0, K[i][0], 0,
                                     0, 0, 1);
  }
  #pragma omp parallel for
  for(size_t i = 0; i<order.size(); ++i) {
    const auto& ord = order[i];
    int root = ord[0].first;
    for(auto it = ++ord.begin(); it != ord.end(); ++it) {
      int p1 = it->first, p2 = it->second;
      shift[p2][p1] = Ks[p2] * Rs[p2] * R_Ts[p1] * Ks[p1].inv();
      shift[p1][p2] = shift[p2][p1].inv();
    }
  }
  for(size_t i = 0; i<pic_num; ++i) shift[i][i] = Mat::eye(3, 3, CV_64FC1);
}
