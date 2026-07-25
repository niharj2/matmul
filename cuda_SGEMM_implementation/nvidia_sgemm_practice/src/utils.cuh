#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---- CUDA ----
void cudaCheck(cudaError_t error, const char *file, int line);
void CudaDeviceInfo();

// ---- Matrix ops ----
void randomize_matrix(float *mat, int N);
void copy_matrix(float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N);
bool verify_matrix(float *mat1, float *mat2, int N);

// ---- Kernel dispatch ----
// kernel_num: 0=cuBLAS, 1=native, 2=coalesce, 3=shared, 4=1d_blocktiling
void test_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C,
                 cublasHandle_t handle);
