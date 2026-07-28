#ifndef memory_coalescing
#define memory_coalescing
#pragma once

#include <cuda_runtime.h>

#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif

void launch_coalesce(
    int M,
    int N,
    int K,
    float* A,
    float* B,
    float* C,
    float alpha,
    float beta
);

void launch_coalesce_alt(
    int M,
    int N,
    int K,
    float* A,
    float* B,
    float* C,
    float alpha,
    float beta
);

__global__ void memory_coalesce(
    int M,
    int N,
    int K,
    float* A,
    float* B,
    float* C,
    float alpha,
    float beta
);


__global__ void memory_coalesce_alternative(
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
