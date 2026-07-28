#ifndef two_d_block_tiling
#define two_d_block_tiling

#pragma once

#include <cuda_runtime.h>


__global__ void two_d_block_tiling_kernel(
    int M, 
    int N, 
    int K, 
    float* A, 
    float* B, 
    float* C, 
    float alpha, 
    float beta
);

#endif
