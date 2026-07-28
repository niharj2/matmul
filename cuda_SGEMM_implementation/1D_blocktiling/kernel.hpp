#pragma once

#include <cuda_runtime.h>

__global__ void one_d_block_tiling_kernel(
    float* A, 
    float* B, 
    float* C, 
    int M, 
    int N, 
    int K, 
    float alpha, 
    float beta
); 
