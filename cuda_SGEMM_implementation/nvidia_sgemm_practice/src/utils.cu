#include <stdio.h>
#include <math.h>
#include "utils.cuh"
#include "kernel.cuh"

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void CudaDeviceInfo() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    printf("Device ID: %d\n"
           "  Number of SMs: %d\n"
           "  Compute Capability: %d.%d\n"
           "  totalGlobalMem: %zuM\n"
           "  sharedMemPerBlock: %zuKB\n"
           "  Warp Size: %d\n",
           deviceId, props.multiProcessorCount, props.major, props.minor,
           props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024, props.warpSize);
}

void randomize_matrix(float *mat, int N) {
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float) (rand() % 5) + 0.01f * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : -tmp;
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, int N) {
    for (int i = 0; i < N; i++) dest[i] = src[i];
}

void print_matrix(const float *A, int M, int N) {
    printf("[");
    for (int i = 0; i < M * N; i++) {
        printf("%5.2f%s", A[i], ((i + 1) % N == 0) ? "" : ", ");
        if ((i + 1) % N == 0 && i + 1 < M * N) printf(";\n");
    }
    printf("]\n");
}

// Combined absolute-OR-relative tolerance: passes if the absolute error is
// tiny (handles near-zero elements where large products cancel) or the
// relative error is tiny. A pure relative check false-fails on cancellation.
// The absolute floor is 1e-2 (not 1e-3) because summation-order drift vs.
// cuBLAS grows with K (more terms accumulated) -- a tighter floor starts
// false-failing large-K sizes on elements that are merely rounding noise.
bool verify_matrix(float *mat1, float *mat2, int N) {
    for (int i = 0; i < N; i++) {
        double diff = fabs((double) mat1[i] - (double) mat2[i]);
        double rel = diff / (fabs((double) mat1[i]) + 1e-6);
        if (diff > 1e-2 && rel > 1e-2) {
            printf("error at %d: cublas=%.5f kernel=%.5f (abs=%.2e rel=%.2e)\n",
                   i, mat1[i], mat2[i], diff, rel);
            return false;
        }
    }
    return true;
}

// ---- Per-kernel launch wrappers ----

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // cuBLAS is column-major; row-major C=A*B is obtained by swapping
    // operands to compute C^T = B^T * A^T in column-major terms.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void test_native(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    native_sgemm<<<grid, block>>>(M, N, K, A, B, C, alpha, beta);
}

void test_coalesce(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    memory_coalesce<<<grid, block>>>(M, K, N, alpha, beta, A, B, C);
}

void test_coalesce_alt(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    memory_coalesce_alternative<<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}

void test_shared(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(CEIL_DIV(N, TILE_WIDTH), CEIL_DIV(M, TILE_WIDTH));
    shared_memory_kernel<<<grid, block>>>(N, M, K, A, B, C, alpha, beta);
}

void test_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C,
                 cublasHandle_t handle) {
    switch (kernel_num) {
        case 0: test_cublas(handle, M, N, K, alpha, A, B, beta, C); break;
        case 1: test_native(M, N, K, alpha, A, B, beta, C); break;
        case 2: test_coalesce(M, N, K, alpha, A, B, beta, C); break;
        case 3: test_coalesce_alt(M, N, K, alpha, A, B, beta, C); break;
        case 4: test_shared(M, N, K, alpha, A, B, beta, C); break;
        default: break;
    }
}
