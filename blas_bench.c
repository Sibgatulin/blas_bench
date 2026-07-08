/*
 * blas_bench.c  --  a minimal, provider-agnostic BLAS/LAPACK benchmark.
 *
 * It calls the standard interfaces:
 *   - cblas_dgemm  : C = alpha*A*B + beta*C   (the BLAS Level-3 workhorse)
 *   - LAPACKE_dgesv: solve A x = b            (a representative LAPACK solve)
 *
 * Because it only uses the *interface*, the exact same binary source can be
 * linked against MKL, OpenBLAS, reference BLAS, etc. Only the link line changes.
 *
 * Metric: GFLOP/s = (floating-point operations) / (seconds) / 1e9.
 *   - dgemm on NxN does 2*N^3 flops (N^3 multiply-adds, each = 2 flops).
 *   - dgesv (LU + solve) does ~ (2/3)N^3 + 2N^2 flops.
 * Higher GFLOP/s = better. Same hardware => the number reflects library quality
 * (kernel vectorisation, blocking, threading) rather than the machine's ceiling.
 *
 * Build examples (only the -l/-I/-L changes between providers):
 *   OpenBLAS:  gcc -O2 blas_bench.c -o blas_bench -lopenblas -llapacke -lm
 *   (via Spack) spack load openblas lapack; then use pkg-config or -L$(...)/lib
 *   MKL:       use Intel's Link Line Advisor for the exact -l flags.
 *
 * Usage: ./blas_bench [N] [reps]
 *   N    : matrix dimension (default 4000)
 *   reps : timed repetitions after one warm-up (default 5); reports median.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/*
 * Header portability: the *function names* (cblas_dgemm, LAPACKE_dgesv) and the
 * enum/constants (CblasRowMajor, LAPACK_ROW_MAJOR) are standardised, so the CODE
 * below never changes between providers. But the header FILE that declares them
 * differs: OpenBLAS/reference ship <cblas.h> + <lapacke.h>, whereas MKL ships
 * everything under <mkl.h> (mkl_cblas.h / mkl_lapacke.h) and has NO bare cblas.h.
 * Verify on your cluster with:  ls $MKLROOT/include | grep -E 'cblas|lapacke'
 * The Spack package passes -DUSE_MKL automatically when the provider is MKL.
 */
#ifdef USE_MKL
#  include <mkl.h>
#else
#  include <cblas.h>
#  include <lapacke.h>
#endif

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_double);
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void fill_random(double *M, long n) {
    for (long i = 0; i < n; ++i) M[i] = (double)rand() / RAND_MAX - 0.5;
}

int main(int argc, char **argv) {
    int N   = (argc > 1) ? atoi(argv[1]) : 4000;
    int reps = (argc > 2) ? atoi(argv[2]) : 5;
    if (N <= 0 || reps <= 0) { fprintf(stderr, "bad args\n"); return 1; }

    long nn = (long)N * N;
    double *A = malloc(nn * sizeof(double));
    double *B = malloc(nn * sizeof(double));
    double *C = malloc(nn * sizeof(double));
    if (!A || !B || !C) { fprintf(stderr, "alloc failed for N=%d\n", N); return 1; }

    srand(12345);
    fill_random(A, nn);
    fill_random(B, nn);
    memset(C, 0, nn * sizeof(double));

    double *times = malloc(reps * sizeof(double));

    /* ---- DGEMM ---- */
    /* warm-up: first call pays one-time costs (paging, MKL ISA dispatch,
       turbo ramp) that we do not want to fold into the timing. */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    for (int r = 0; r < reps; ++r) {
        double t0 = wall_seconds();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        times[r] = wall_seconds() - t0;
    }
    double t_gemm = median(times, reps);
    double gflop_gemm = 2.0 * (double)N * N * N / t_gemm / 1e9;

    /* ---- DGESV (solve A x = b, A gets overwritten by its LU each time) ---- */
    double *Asolve = malloc(nn * sizeof(double));
    double *b      = malloc((long)N * sizeof(double));
    int    *ipiv   = malloc((long)N * sizeof(int));
    double t_solve = NAN, gflop_solve = NAN;
    if (Asolve && b && ipiv) {
        for (int r = 0; r < reps; ++r) {
            memcpy(Asolve, A, nn * sizeof(double));
            fill_random(b, N);
            double t0 = wall_seconds();
            LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, Asolve, N, ipiv, b, 1);
            times[r] = wall_seconds() - t0;
        }
        t_solve = median(times, reps);
        gflop_solve = ((2.0 / 3.0) * N * N * N + 2.0 * N * N) / t_solve / 1e9;
    }

    /* Machine-readable single line + a human line. Easy to grep/collate later. */
    printf("RESULT N=%d reps=%d dgemm_s=%.4f dgemm_gflops=%.2f "
           "dgesv_s=%.4f dgesv_gflops=%.2f\n",
           N, reps, t_gemm, gflop_gemm, t_solve, gflop_solve);
    printf("  DGEMM  N=%d : %.3f s  -> %.1f GFLOP/s\n", N, t_gemm, gflop_gemm);
    printf("  DGESV  N=%d : %.3f s  -> %.1f GFLOP/s\n", N, t_solve, gflop_solve);

    free(A); free(B); free(C); free(times);
    free(Asolve); free(b); free(ipiv);
    return 0;
}
