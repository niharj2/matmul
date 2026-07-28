#pragma once

#include <cuda_runtime.h>

__global__ void MatrixMultiplicationKernel(
    int M, 
    int N, 
    int K, 
    float* A, 
    float* B, 
    float* C, 
    float alpha, 
    float beta
);
