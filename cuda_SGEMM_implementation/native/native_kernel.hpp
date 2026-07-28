#ifndef native_kernel 
#define native_kernel

#pragma once

#include <cuda_runtime.h>

__global__ void native_sgemm(
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
