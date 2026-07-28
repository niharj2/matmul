// General code to launch the kernels

#include <cstdio>
#include <cstdlib>
#include "run.hpp"

#include "1D_blocktiling/one_d_block_tiling.hpp"
#include "2D_blocktiling/two_d_block_tiling.hpp"
#include "shared_memory_cache_blocking/shared_memory.hpp"
#include "memory_coalescing/memory_coalescing.hpp"
#include "native/native_kernel.hpp"

// Will be using the 2D block tiling kernel because its the most effective right now

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void MatMul(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    float* Ad;
    float* Bd;
    float* Cd;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&Ad, size_A));
    CUDA_CHECK(cudaMalloc((void**)&Bd, size_B));
    CUDA_CHECK(cudaMalloc((void**)&Cd, size_C));

    CUDA_CHECK(cudaMemcpy(Ad, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Bd, B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Cd, C, size_C, cudaMemcpyHostToDevice));

    launch_2d(M, N, K, Ad, Bd, Cd, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, Cd, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(Ad));
    CUDA_CHECK(cudaFree(Bd));
    CUDA_CHECK(cudaFree(Cd));
}
