#ifndef shared_memory
#define shared_memory
#pragma once

#include <cuda_runtime.h>

__global__ void shared_memory_kernel_same_size(
    int N, 
    float* A, 
    float* B, 
    float* C, 
    float alpha,
    float beta
);


__global__ void shared_memory_kernel(
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
