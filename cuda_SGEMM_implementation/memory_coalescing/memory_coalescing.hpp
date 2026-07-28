#ifndef memory_coalescing
#define memory_coalescing
#pragma once

#include <cuda_runtime.h>

__global__ void memory_coalesce(
    int M, 
    int K, 
    int N, 
    float alpha, 
    float beta, 
    float* A, 
    float* B, 
    float* C
);


__global__ void memory_coalesce_alternative(
    int M, 
    int N, 
    int K, 
    float alpha, 
    float beta, 
    float* A, 
    float* B, 
    float*C
); 

#endif
