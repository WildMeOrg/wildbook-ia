// g++ -Wall -Wextra sver.cpp -lopencv_core -shared -fPIC -O2 -ffast-math -o sver.so

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
#include <vector>

// References:
// http://stackoverflow.com/questions/487108/how-to-supress-specific-warnings-in-g
// suppress unused warnings of specific variables
#define MARKUSED(X)  ((void)(&(X)))

//#define RUNTIME_BOUNDS_CHECKING
#ifdef RUNTIME_BOUNDS_CHECKING
#define CHECK_FM_BOUNDS(fm, fm_len, kpts1_len, kpts2_len) \
    for(size_t fm_ind = 0; fm_ind < fm_len; fm_ind += 2) { \
        if((fm[fm_ind] >= kpts1_len) || (fm[fm_ind + 1] >= kpts2_len)) { \
            puts("CHECK_FM_BOUNDS: bad fm indexes"); \
            return; \
        } \
    }
#else
#define CHECK_FM_BOUNDS(fm, fm_len, kpts1_len, kpts2_len)
#endif

using cv::Matx;
using std::vector;

// adapted from hesaff's helpers.h
#ifndef M_TAU
#define M_TAU 6.28318
#endif
template <class T> T ensure_0toTau(T x)
{
    if(x < 0) { return ensure_0toTau(x+M_TAU); }
    else if(x >= M_TAU) { return ensure_0toTau(x-M_TAU); }
    else { return x; }
}

template<typename T> Matx<T, 3, 3> get_invV_mat(T x, T y, T a, T c, T d, T theta) {
    T ct = (T)cos(theta), st = (T)sin(theta);
    // https://github.com/aweinstock314/haskell-stuff/blob/master/ExpressionSimplifier.hs
    return Matx<T, 3, 3>(
             a*ct,      a*(-st),   x,
        c*ct+d*st, c*(-st)+d*ct,   y,
              0.0,        0.0, 1.0);
}

void debug_print_kpts(double* kpts, size_t kpts_len) {
    for(size_t i=0; i<kpts_len*6; i+=6) {
        printf("kpts[%lu]: [%f, %f, %f, %f, %f, %f]\n", i/6,
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

template<typename T> inline T xy_distance(const Matx<T, 3, 3>& kpt1, const Matx<T, 3, 3>& kpt2) {
    // ktool.get_invVR_mats_xys
    T x1 = kpt1(0, 2), y1 = kpt1(1, 2);
    T x2 = kpt2(0, 2), y2 = kpt2(1, 2);
    // ltool.L2_sqrd
    T dx = x2-x1, dy = y2-y1;
    return dx*dx + dy*dy;
}

template<typename T> inline T det_distance(const Matx<T, 3, 3>& kpt1, const Matx<T, 3, 3>& kpt2) {
    // ktool.get_invVR_mats_sqrd_scale
    T a1 = kpt1(0, 0), b1 = kpt1(0, 1), c1 = kpt1(1, 0), d1 = kpt1(1, 1);
    T a2 = kpt2(0, 0), b2 = kpt2(0, 1), c2 = kpt2(1, 0), d2 = kpt2(1, 1);
    T det1 = a1*d1 - b1*c1, det2 = a2*d2 - b2*c2;
    // ltool.det_distance
    T dist = det1/det2;
    if(dist < 1) { dist = 1/dist; }
    return dist;
}

template<typename T> inline T ori_distance(const Matx<T, 3, 3>& kpt1, const Matx<T, 3, 3>& kpt2) {
    // ktool.get_invVR_mats_oris
    T a1 = kpt1(0, 0), b1 = kpt1(0, 1);
    T a2 = kpt2(0, 0), b2 = kpt2(0, 1);
    T ori1 = ensure_0toTau(-atan2(b1, a1));
    T ori2 = ensure_0toTau(-atan2(b2, a2));
    // ltool.ori_distance
    T delta = fabs(ori1 - ori2);
    delta = ensure_0toTau(delta);
    return std::min(delta, M_TAU-delta);
}

template<typename T> inline Matx<T, 3, 3> get_Aff_mat(const Matx<T, 3, 3>& invVR1_m,
                                                      const Matx<T, 3, 3>& invVR2_m) {
    
    const Matx<double, 3, 3> V1_m = invVR1_m.inv();
    const Matx<double, 3, 3> Aff_mat = invVR2_m * V1_m;
    return Aff_mat;
}

extern "C" {
    void get_affine_inliers(double* kpts1, size_t kpts1_len,
                    double* kpts2, size_t kpts2_len,
                    size_t* fm, size_t fm_len,
                    double xy_thresh_sqrd, double scale_thresh_sqrd, double ori_thresh,
                    // memory is expected to by allocated by the caller (i.e. via numpy.empty)
                    bool* out_inlier_flags, double* out_errors_list, double* out_matrices_list)
    {

        MARKUSED(kpts1_len);
        MARKUSED(kpts2_len);
        CHECK_FM_BOUNDS(fm, fm_len, kpts1_len, kpts2_len);
// remove some redundancy in a possibly-ugly way
#define SETUP_RELEVANT_VARIABLES \
double* kpt1 = &kpts1[6*fm[fm_ind+0]]; \
double* kpt2 = &kpts2[6*fm[fm_ind+1]]; \
Matx<double, 3, 3> invVR1_m = get_invV_mat( \
    kpt1[0], kpt1[1], kpt1[2], \
    kpt1[3], kpt1[4], kpt1[5]); \
Matx<double, 3, 3> invVR2_m = get_invV_mat( \
    kpt2[0], kpt2[1], kpt2[2], \
    kpt2[3], kpt2[4], kpt2[5]);
        //vector<Matx<double, 3, 3> > Aff_mats;
// MATRIX_REF(i) should be the same as Aff_mats[i], but 
//  directly operating on the numpy-allocated memory
//   (less allocation == faster code)
#define MATRIX_REF(i) (*((i)+((Matx<double, 3, 3>*)out_matrices_list)))
        //vector<vector<double> > xy_errs, scale_errs, ori_errs;
        for(size_t fm_ind = 0; fm_ind < fm_len; fm_ind += 2) {
            SETUP_RELEVANT_VARIABLES
            //Aff_mats.push_back(get_Aff_mat(invVR1_m, invVR2_m));
            MATRIX_REF(fm_ind/2) = get_Aff_mat(invVR1_m, invVR2_m);
        }
        const size_t num_matches = fm_len/2;
        for(size_t i = 0; i < num_matches; i++) {
            //xy_errs.push_back(vector<double>());
            //scale_errs.push_back(vector<double>());
            //ori_errs.push_back(vector<double>());
            for(size_t fm_ind = 0; fm_ind < fm_len; fm_ind += 2) {
                SETUP_RELEVANT_VARIABLES
                Matx<double, 3, 3> Aff_mat = MATRIX_REF(i);
                // _test_hypothesis_inliers
                Matx<double, 3, 3> invVR1_mt = Aff_mat * invVR1_m;
                double xy_err = xy_distance(invVR1_mt, invVR2_m);
                double scale_err = det_distance(invVR1_mt, invVR2_m);
                double ori_err = ori_distance(invVR1_mt, invVR2_m);
                //xy_errs[i].push_back(xy_err);
                //scale_errs[i].push_back(scale_err);
                //ori_errs[i].push_back(ori_err);
// poke the error values directly into the output array with pointer voodoo to 
//  avoid intermediate allocations (the explicit structure is shown by the 
//   commented xy_errs, scale_errs, and ori_errs variables).
#define PACKED_INSERT(OFFSET, VAR) \
*(out_errors_list+(num_matches*3*i)+(num_matches*(OFFSET))+(fm_ind/2)) = (VAR)
                PACKED_INSERT(0, xy_err);
                PACKED_INSERT(1, ori_err);
                PACKED_INSERT(2, scale_err);
#undef PACKED_INSERT
                bool is_inlier = (xy_err < xy_thresh_sqrd) &&
                                 (scale_err < scale_thresh_sqrd) &&
                                 (ori_err < ori_thresh);
                *(out_inlier_flags+(num_matches*i)+(fm_ind/2)) = is_inlier;
                //printf("errs[%u][%u]: %f, %f, %f\n", fm_ind, i, xy_err, scale_err, ori_err);
            }
        }
#undef MATRIX_REF
#undef SETUP_RELEVANT_VARIABLES
/*
#define SHOW_ERRVEC(vec) \
for(size_t i = 0; i < vec.size(); i++) { \
    putchar('['); \
    for(size_t j = 0; j < vec[i].size(); j++) { \
        printf("%f, ", vec[i][j]); \
    } \
    puts("]"); \
}
        puts("-----");
        SHOW_ERRVEC(xy_errs)
        puts("-----");
        SHOW_ERRVEC(scale_errs)
        puts("-----");
        SHOW_ERRVEC(ori_errs)
        puts("-----");
#undef SHOW_ERRVEC
*/
        // Code for copying Aff_mats into the output is redundant now 
        //  that the output is operated on directly (via MATRIX_REF)
        /*
        //printf("%lu\n", Aff_mats.size());
        for(size_t i = 0; i < Aff_mats.size(); i++) {
            const size_t mat_size = 3*3*sizeof(double);
            //char msg[] = {'M', 'a', 't', 0x30+i%10, 0};
            //debug_print_mat3x3(msg, Aff_mats[i]);
            double* dest = (out_matrices_list+(3*3*i));
            //char* destc = (char*)(out_matrices_list+(3*3*i));
            //printf("%x\n", dest);
            //printf("before: "); for(size_t j=0; j < 9; j++) {printf("%f ", *(dest+j)); }
            //printf("\nbefore: "); for(size_t j=0; j < mat_size; j+=8) {printf("0x%08x ", *(destc+j)); }
            memcpy(dest, &Aff_mats[i], mat_size);
            //printf("\nafter: "); for(size_t j=0; j < 9; j++) {printf("%f ", *(dest+j)); }
            //printf("\nafter: "); for(size_t j=0; j < mat_size; j+=8) {printf("0x%08x ", *(destc+j)); }
            //puts("\n");
        }
        */
    }

    void hello_world() {
        puts("Hello from C++!");
    }
}
