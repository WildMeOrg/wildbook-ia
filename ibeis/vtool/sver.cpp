// g++ sver.cpp -lopencv_core -shared -fPIC -o sver.so

/*
keypoints of form (x, y, a, c, d, theta)
each represents "invVR" matrix = 
/     \     /                         \
|a 0 x|     |cos(theta) -sin(theta)  0|
|c d y|  *  |sin(theta)  cos(theta)  0|
|0 0 1|     |         0           0  1|
\     /     \                         /

"kpts" parameters are keypoint lists
-----
fm == feature_matches :: [(Int, Int)]
indices into kpts{1,2} indicating a match
*/
#include <cmath>
#include <cstdio>
#include <opencv2/core/core.hpp>

using cv::Matx;

template<typename _Tp> Matx<_Tp, 3, 3> get_invV_mat(_Tp a, _Tp c, _Tp d, _Tp x, _Tp y, _Tp theta) {
    _Tp ct = (_Tp)cos(theta), st = (_Tp)sin(theta);
    // https://github.com/aweinstock314/haskell-stuff/blob/master/ExpressionSimplifier.hs
    return Matx<_Tp, 3, 3>(
             a*ct,    a*(-st),   x,
        c*ct+d*st, c*(-st)+d*ct,   y,
              0.0,        0.0, 1.0);
}

void debug_print_kpts(double* kpts, size_t kpts_len) {
    for(size_t i=0; i<kpts_len*6; i+=6) {
        printf("kpts[%u]: [%f, %f, %f, %f, %f, %f]\n", i/6,
            kpts[i+0], kpts[i+1], kpts[i+2],
            kpts[i+3], kpts[i+4], kpts[i+5]);
    }
}

void debug_print_mat3x3(const char* name, const Matx<double, 3, 3>& mat) {
    printf("%s: [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]\n", name,
        mat(0, 0), mat(0, 1), mat(0, 2),
        mat(1, 0), mat(1, 1), mat(1, 2),
        mat(2, 0), mat(2, 1), mat(2, 2));
}

template<typename _Tp> _Tp xy_distance(Matx<_Tp, 3, 3> kpt1, Matx<_Tp, 3, 3> kpt2) {
    _Tp x1 = kpt1(0, 2), y1 = kpt1(1, 2);
    _Tp x2 = kpt2(0, 2), y2 = kpt2(1, 2);
    _Tp dx = x2-x1, dy = y2-y1;
    return sqrt(dx*dx + dy*dy);
}

extern "C" {
    void get_affine_inliers(double* kpts1, size_t kpts1_len,
                    double* kpts2, size_t kpts2_len,
                    size_t* fm, size_t fm_len,
                    double xy_thresh_sqrd, double scale_thresh_sqrd, double ori_thresh)
    {
        for(size_t fm_ind = 0; fm_ind < fm_len; fm_ind += 2) {
            //printf("fm[[%u, %u]] == [%u, %u]\n", fm_ind, fm_ind+1, fm[fm_ind], fm[fm_ind+1]);
            double* kpt1 = &kpts1[6*fm[fm_ind+0]];
            double* kpt2 = &kpts2[6*fm[fm_ind+1]];
            Matx<double, 3, 3> invVR1_m = get_invV_mat(
                kpt1[0], kpt1[1], kpt1[2],
                kpt1[3], kpt1[4], kpt1[5]);
            Matx<double, 3, 3> invVR2_m = get_invV_mat(
                kpt2[0], kpt2[1], kpt2[2],
                kpt2[3], kpt2[4], kpt2[5]);
            Matx<double, 3, 3> V1_m = invVR1_m.inv();
            Matx<double, 3, 3> Aff_mat = invVR2_m * V1_m;
            debug_print_mat3x3("Aff_mat", Aff_mat);
        }
    }

    void hello_world() {
        puts("Hello from C++!");
    }
}
