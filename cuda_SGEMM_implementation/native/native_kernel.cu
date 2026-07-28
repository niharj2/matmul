#include "native_kernel.hpp"
// Implementing a native CUDA SGEMM kernel. SGEMM performs C = αAB+βC at single (=32b) precision.

// Dimensions of matrix A would be M * K
// Dimensooms of matrix B would be K * N 
// Therefore, dimensions of matric C would be M * N

__global__ void native_sgemm(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x; 

    // makes sure we aren't going out of bounds

    if (row < M && col < N) {
        float sum = 0; 
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        
        // Using N instead of K, because the stride becomes N since that's the updated dimension of the matrix 
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_native(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    dim3 block(32, 32);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    native_sgemm<<<grid, block>>>(M, N, K, A, B, C, alpha, beta);
}
