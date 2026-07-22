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
 * Every timing is accompanied by a correctness check (see "verification"
 * below) and by provenance about which library actually got loaded, because a
 * benchmark that reports a healthy GFLOP/s while returning garbage -- or while
 * silently running single-threaded -- is worse than no benchmark at all.
 *
 * Build examples (only the -l/-I/-L changes between providers):
 *   OpenBLAS:  gcc -O2 blas_bench.c -o blas_bench -lopenblas -llapacke -lm -ldl
 *   MKL:       use Intel's Link Line Advisor for the exact -l flags.
 * In practice: let the Spack recipe build it, so the provider-specific link
 * line is derived from the concretised spec rather than typed by hand.
 *
 * Usage: ./blas_bench [N] [reps]
 *   N    : matrix dimension (default 4000)
 *   reps : timed repetitions after one warm-up (default 5); reports median.
 */

/* RTLD_DEFAULT (used for the runtime provenance probes) is a GNU extension. */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <dlfcn.h>

/*
 * Header portability: the *function names* (cblas_dgemm, LAPACKE_dgesv) and the
 * enum/constants (CblasColMajor, LAPACK_COL_MAJOR) are standardised, so the CODE
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

/*
 * Build-time provenance, injected by the Spack recipe so that each output line
 * names the spec that produced it. Defaulted here so the file still compiles
 * with a bare `gcc blas_bench.c`.
 */
#ifndef BLAS_PROVIDER
#  define BLAS_PROVIDER "unknown"
#endif
#ifndef SPEC_HASH
#  define SPEC_HASH "none"
#endif

/*
 * Verification thresholds, expressed as *scaled* residuals -- i.e. the error
 * divided by the error that correct floating-point arithmetic would already
 * incur at this problem size. A correct implementation lands at O(1); we allow
 * three orders of magnitude of slack. Garbage from a mismatched threading layer
 * lands at 1e10 and up, so the exact threshold is not delicate.
 */
#define RESID_TOL 1.0e3

/* Number of entries of C spot-checked against a hand-rolled dot product. */
#define GEMM_SAMPLES 16

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

static double max_abs(const double *M, long n) {
    double m = 0.0;
    for (long i = 0; i < n; ++i) { double a = fabs(M[i]); if (a > m) m = a; }
    return m;
}

/*
 * Runtime provenance. These probes ask the *process* what it actually loaded,
 * rather than trusting what we think we linked. dlsym(RTLD_DEFAULT, ...) walks
 * the already-loaded objects: no dlopen, no provider-specific header, and no
 * build-flag branching, so one code path covers MKL, OpenBLAS and anything else.
 *
 * The cast from void* to a function pointer is not strictly ISO C, but it is
 * exactly what POSIX specifies dlsym for, and every relevant compiler accepts it.
 */
static int omp_max_threads(void) {
    /*
     * Resolves only if an OpenMP runtime is in the image. This is THE check
     * that distinguishes a real `threads=openmp` build from one where the
     * threading layer silently failed to be linked: a sequential build returns
     * -1 here, and it should never return -1 for a threaded one.
     */
    int (*fn)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "omp_get_max_threads");
    return fn ? fn() : -1;
}

static void provider_version(char *buf, size_t n) {
    void (*mkl)(char *, int)  = (void (*)(char *, int))dlsym(RTLD_DEFAULT, "mkl_get_version_string");
    char *(*ob)(void)         = (char *(*)(void))dlsym(RTLD_DEFAULT, "openblas_get_config");
    if (mkl) { mkl(buf, (int)n); return; }
    if (ob)  { snprintf(buf, n, "%s", ob()); return; }
    snprintf(buf, n, "unknown");
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
    if (!times) { fprintf(stderr, "alloc failed\n"); return 1; }

    /*
     * Everything below is COLUMN-major. Not a stylistic choice: LAPACKE's
     * row-major entry points do not reach a row-major LAPACK (there is none) --
     * they transpose-copy the whole matrix in and out around the call. At
     * N=8000 that is two extra 512 MB copies *inside the timed region*, so a
     * row-major dgesv partly benchmarks LAPACKE's glue rather than the solver.
     * cblas_dgemm is honest about row-major (it just swaps the operands), but
     * we keep one convention throughout so the residual checks below are not
     * quietly checking a different layout than the calls they verify.
     * Since A, B and C are square with leading dimension N, the *arguments*
     * happen to be unchanged -- except dgesv's ldb, which goes 1 -> N.
     */

    /* ---- DGEMM ---- */
    /* warm-up: first call pays one-time costs (paging, MKL ISA dispatch,
       turbo ramp) that we do not want to fold into the timing. */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    for (int r = 0; r < reps; ++r) {
        double t0 = wall_seconds();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, B, N, 0.0, C, N);
        times[r] = wall_seconds() - t0;
    }
    double t_gemm = median(times, reps);
    double gflop_gemm = 2.0 * (double)N * N * N / t_gemm / 1e9;

    /*
     * dgemm verification, OUTSIDE the timed region: recompute a handful of
     * entries of C as plain dot products and compare. Checking all N^2 entries
     * would cost as much as the benchmark; a spot check is enough, because the
     * failure mode we are guarding against (a mismatched threading layer, a
     * half-initialised library) corrupts results wholesale, not in single cells.
     * The error is scaled by N*eps*max|A|*max|B|, the size of the rounding error
     * a correct dot product of this length already carries, so a healthy result
     * is O(1) regardless of N.
     */
    double amax = max_abs(A, nn), bmax = max_abs(B, nn);
    double gemm_scale = (double)N * DBL_EPSILON * amax * bmax;
    double gemm_err = 0.0;
    unsigned long lcg = 88172645463325252ULL;  /* deterministic sample sites */
    for (int s = 0; s < GEMM_SAMPLES; ++s) {
        lcg ^= lcg << 13; lcg ^= lcg >> 7; lcg ^= lcg << 17;
        int i = (int)(lcg % (unsigned long)N);
        lcg ^= lcg << 13; lcg ^= lcg >> 7; lcg ^= lcg << 17;
        int j = (int)(lcg % (unsigned long)N);
        double ref = 0.0;
        for (int k = 0; k < N; ++k) ref += A[i + (long)k * N] * B[k + (long)j * N];
        double e = fabs(C[i + (long)j * N] - ref) / gemm_scale;
        if (e > gemm_err) gemm_err = e;
    }

    /* ---- DGESV (solve A x = b, A gets overwritten by its LU each time) ---- */
    double *Asolve = malloc(nn * sizeof(double));
    double *b      = malloc((long)N * sizeof(double));
    double *bsave  = malloc((long)N * sizeof(double));
    int    *ipiv   = malloc((long)N * sizeof(int));
    double t_solve = NAN, gflop_solve = NAN, gesv_err = NAN;
    int info = 0;
    if (Asolve && b && bsave && ipiv) {
        for (int r = 0; r < reps; ++r) {
            memcpy(Asolve, A, nn * sizeof(double));
            fill_random(b, N);
            memcpy(bsave, b, (long)N * sizeof(double));  /* dgesv overwrites b with x */
            double t0 = wall_seconds();
            info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, 1, Asolve, N, ipiv, b, N);
            times[r] = wall_seconds() - t0;
        }
        t_solve = median(times, reps);
        gflop_solve = ((2.0 / 3.0) * N * N * N + 2.0 * N * N) / t_solve / 1e9;

        /*
         * Backward-error check on the last solve, outside the timed region.
         * A is untouched by dgesv (it works on the Asolve copy), so we can form
         * the residual r = A*x - b with one dgemv and scale it the way LAPACK's
         * own test suite does:  ||Ax-b||_inf / (||A||_inf ||x||_inf N eps).
         * A correct solve is O(1) here even for an ill-conditioned A -- this
         * tests the *solver*, not the conditioning of a random matrix.
         */
        double *rvec = malloc((long)N * sizeof(double));
        double *rows = calloc((size_t)N, sizeof(double));
        if (rvec && rows) {
            memcpy(rvec, bsave, (long)N * sizeof(double));
            cblas_dgemv(CblasColMajor, CblasNoTrans, N, N,
                        1.0, A, N, b, 1, -1.0, rvec, 1);   /* rvec = A*x - b */
            /* ||A||_inf = max row sum; accumulate column-wise to stay sequential
               in memory (a naive row walk strides 64 kB per element at N=8000). */
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < N; ++i) rows[i] += fabs(A[i + (long)j * N]);
            double anorm = 0.0, xnorm = 0.0, rnorm = 0.0;
            for (int i = 0; i < N; ++i) {
                if (rows[i] > anorm) anorm = rows[i];
                if (fabs(b[i]) > xnorm) xnorm = fabs(b[i]);
                if (fabs(rvec[i]) > rnorm) rnorm = fabs(rvec[i]);
            }
            gesv_err = rnorm / (anorm * xnorm * (double)N * DBL_EPSILON);
        }
        free(rvec); free(rows);
    }

    int resid_ok = (gemm_err <= RESID_TOL) && (info == 0) &&
                   (gesv_err == gesv_err) && (gesv_err <= RESID_TOL);

    /* ---- provenance ---- */
    int omp_threads = omp_max_threads();
    char version[512];
    provider_version(version, sizeof version);

    /*
     * Two machine-readable lines. RESULT is strictly space-separated key=value
     * with no embedded spaces in any value, so it survives `grep ^RESULT | tr`
     * straight into a CSV. Version strings contain spaces, so they live on their
     * own INFO line instead of contaminating that contract.
     */
    printf("RESULT provider=%s hash=%s omp_threads=%d N=%d reps=%d "
           "dgemm_s=%.4f dgemm_gflops=%.2f dgesv_s=%.4f dgesv_gflops=%.2f "
           "dgemm_err=%.2e dgesv_err=%.2e info=%d resid_ok=%d\n",
           BLAS_PROVIDER, SPEC_HASH, omp_threads, N, reps,
           t_gemm, gflop_gemm, t_solve, gflop_solve,
           gemm_err, gesv_err, info, resid_ok);
    printf("INFO version=%s\n", version);
    printf("  DGEMM  N=%d : %.3f s  -> %.1f GFLOP/s   (scaled err %.2e)\n",
           N, t_gemm, gflop_gemm, gemm_err);
    printf("  DGESV  N=%d : %.3f s  -> %.1f GFLOP/s   (scaled resid %.2e, info=%d)\n",
           N, t_solve, gflop_solve, gesv_err, info);
    if (!resid_ok)
        fprintf(stderr, "FAIL: verification failed -- timings above are meaningless\n");

    free(A); free(B); free(C); free(times);
    free(Asolve); free(b); free(bsave); free(ipiv);
    return resid_ok ? 0 : 2;
}
