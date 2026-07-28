// SGEMM test + benchmark harness.
//
// Computes C = alpha*A*B + beta*C  (A: MxK, B: KxN, C: MxN, row-major)
// - Verifies each kernel against a cuBLAS reference.
// - Times each kernel with CUDA events and reports GFLOPS.
//
// Build:   make            (from this directory)
// Run:     ./sgemm [M] [N] [K] [kernel]
//   kernel in {native, coalesce, coalesce_alt, shared, cublas, all}  (default: all)
//
// TILE_WIDTH / BLOCKSIZE can be overridden at compile time, e.g.
//   nvcc -DTILE_WIDTH=16 ...   (used for the tile-size sweep experiment)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---- Tunables the kernels depend on -------------------------------------
#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---- Pull in the kernels under test (the real source of truth) ----------
#include "../native/native_kernel.cu"
#include "../memory_coalescing/memory_coalescing_kernel.cu"
#include "../shared_memory_cache_blocking/shared_memory_kernel.cu"

// ---- Error-checking helpers ---------------------------------------------
#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t err = (x);                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(x)                                                      \
    do {                                                                     \
        cublasStatus_t st = (x);                                            \
        if (st != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", st, __FILE__,     \
                    __LINE__);                                              \
            exit(EXIT_FAILURE);                                            \
        }                                                                    \
    } while (0)

// ---- Kernel launch wrappers (hide the differing signatures) -------------
// Each takes the device pointers plus dims/scalars and launches the kernel.

static void launch_native(int M, int N, int K, float alpha, float beta,
                          const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));  // x->col (N), y->row (M)
    native_sgemm<<<grid, block>>>(M, N, K, (float*)A, (float*)B, C, alpha, beta);
}

static void launch_coalesce(int M, int N, int K, float alpha, float beta,
                            const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    memory_coalesce<<<grid, block>>>(M, N, K, (float*)A, (float*)B, C, alpha, beta);
}

static void launch_coalesce_alt(int M, int N, int K, float alpha, float beta,
                                const float* A, const float* B, float* C) {
    // 1D block of BLOCKSIZE*BLOCKSIZE threads.
    // In the kernel: grid.x maps to rows (M), grid.y maps to cols (N).
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    memory_coalesce_alternative<<<grid, block>>>(M, N, K, (float*)A, (float*)B,
                                                 C, alpha, beta);
}

static void launch_shared(int M, int N, int K, float alpha, float beta,
                          const float* A, const float* B, float* C) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(CEIL_DIV(N, TILE_WIDTH), CEIL_DIV(M, TILE_WIDTH));
    shared_memory_kernel<<<grid, block>>>(M, N, K, (float*)A, (float*)B, C,
                                          alpha, beta);
}

// cuBLAS reference. cuBLAS is column-major; a row-major C = A*B is obtained
// by computing, in column-major terms, C^T = B^T * A^T via swapping operands.
static cublasHandle_t g_handle = nullptr;
static void launch_cublas(int M, int N, int K, float alpha, float beta,
                          const float* A, const float* B, float* C) {
    CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             B, N,      // B is NxK when viewed col-major
                             A, K,      // A is KxM when viewed col-major
                             &beta, C, N));
}

typedef void (*LaunchFn)(int, int, int, float, float,
                         const float*, const float*, float*);

struct Kernel {
    const char* name;
    LaunchFn    fn;
};

// ---- Utilities -----------------------------------------------------------
static void fill_random(std::vector<float>& v) {
    for (auto& x : v) x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Max effective error under a combined absolute-OR-relative tolerance.
// An element is "close enough" if EITHER its absolute error is tiny (handles
// near-zero results where big products cancel) OR its relative error is tiny.
// We report min(rel, abs/ABS_TOL): an element passes iff this stays < 1e-2.
static const double ABS_TOL = 1e-2;  // absolute-error floor (grows with K, see practice harness)
static double max_err(const std::vector<float>& test,
                      const std::vector<float>& ref) {
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = fabs((double)test[i] - (double)ref[i]);
        double rel  = diff / (fabs(ref[i]) + 1e-6);
        // Scale absolute error so that diff == ABS_TOL reads as 1e-2, matching
        // the rel-error cutoff; take whichever measure is more forgiving.
        double abs_scaled = (diff / ABS_TOL) * 1e-2;
        double err = rel < abs_scaled ? rel : abs_scaled;
        if (err > worst) worst = err;
    }
    return worst;
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    int K = argc > 3 ? atoi(argv[3]) : 1024;
    std::string which = argc > 4 ? argv[4] : "all";

    const float alpha = 1.0f;
    const float beta  = 0.5f;   // exercise the beta*C path too
    const int   WARMUP = 3;
    const int   ITERS  = 20;

    printf("SGEMM  M=%d N=%d K=%d  (TILE_WIDTH=%d, BLOCKSIZE=%d)\n",
           M, N, K, TILE_WIDTH, BLOCKSIZE);
    printf("alpha=%.2f beta=%.2f  |  verifying vs cuBLAS, tol=1e-2\n\n",
           alpha, beta);

    // Host data.
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC0((size_t)M * N);
    srand(0);
    fill_random(hA); fill_random(hB); fill_random(hC0);

    // Device buffers.
    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA,   sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&dB,   sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&dC,   sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&dRef, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate(&g_handle));

    // Reference result from cuBLAS (starting from the same C0).
    CUDA_CHECK(cudaMemcpy(dRef, hC0.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
    launch_cublas(M, N, K, alpha, beta, dA, dB, dRef);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    std::vector<Kernel> kernels = {
        {"native",       launch_native},
        {"coalesce",     launch_coalesce},
        {"coalesce_alt", launch_coalesce_alt},
        {"shared",       launch_shared},
        {"cublas",       launch_cublas},
    };

    printf("%-14s %12s %12s %14s   %s\n",
           "kernel", "time(ms)", "GFLOPS", "max_err", "verdict");
    printf("--------------------------------------------------------------------\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const double flop = 2.0 * M * N * K;

    for (const auto& k : kernels) {
        if (which != "all" && which != k.name) continue;

        // Correctness: run once from a fresh copy of C0, compare to ref.
        CUDA_CHECK(cudaMemcpy(dC, hC0.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
        k.fn(M, N, K, alpha, beta, dA, dB, dC);
        cudaError_t launchErr = cudaGetLastError();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out((size_t)M * N);
        CUDA_CHECK(cudaMemcpy(out.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        double rel = (launchErr == cudaSuccess) ? max_err(out, ref) : INFINITY;

        // Timing.
        for (int i = 0; i < WARMUP; ++i)
            k.fn(M, N, K, alpha, beta, dA, dB, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i)
            k.fn(M, N, K, alpha, beta, dA, dB, dC);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= ITERS;
        double gflops = flop / (ms * 1e6);

        const char* verdict = (launchErr != cudaSuccess) ? "LAUNCH ERR"
                            : (rel < 1e-2 ? "PASS" : "FAIL");
        printf("%-14s %12.4f %12.1f %14.2e   %s\n",
               k.name, ms, gflops, rel, verdict);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cublasDestroy(g_handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    return 0;
}
