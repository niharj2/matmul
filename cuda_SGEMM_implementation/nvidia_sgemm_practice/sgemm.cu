// Adapted from wangzyon/NVIDIA_SGEMM_PRACTICE's sgemm.cu, wired to test the
// kernels in cuda_SGEMM_implementation instead of that repo's mysgemm_v*.
//
// kernel_num: 0=cuBLAS, 1=native, 2=coalesce, 3=coalesce_alt, 4=shared
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Please select a kernel (0=cublas, 1=native, 2=coalesce, 3=coalesce_alt, 4=shared).\n");
        exit(EXIT_FAILURE);
    }

    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 4) {
        printf("Please enter a valid kernel number (0-4).\n");
        exit(EXIT_FAILURE);
    }
    printf("Select kernel %d.\n", kernel_num);

    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    }

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Matrix sizes: 256, 512, ..., 6144 (matches the original repo's sweep).
    int size_len = 24;
    int SIZE[size_len];
    for (int i = 0; i < size_len; i++)
        SIZE[i] = 256 * (i + 1);

    int m, n, k, max_size;
    max_size = SIZE[size_len - 1];
    printf("max_size=%d\n", max_size);

    float alpha = 1.0f, beta = 0.0f; // C = alpha*A*B + beta*C

    float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL;
    float *dA = NULL, *dB = NULL, *dC = NULL, *dC_ref = NULL;

    A = (float *) malloc(sizeof(float) * max_size * max_size);
    B = (float *) malloc(sizeof(float) * max_size * max_size);
    C = (float *) malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *) malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);
    copy_matrix(C, C_ref, max_size * max_size);

    cudaCheck(cudaMalloc((void **) &dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeat_times = 10;
    for (int i = 0; i < size_len; i++) {
        m = n = k = SIZE[i];
        printf("m=n=k=%d\n", m);

        // Verify against cuBLAS, and warm up before timing.
        if (kernel_num != 0) {
            test_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
            cudaDeviceSynchronize();
            cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            if (!verify_matrix(C_ref, C, m * n)) {
                printf("Failed to pass the correctness verification against cuBLAS. Exited.\n");
                exit(EXIT_FAILURE);
            }
        }
        cudaDeviceSynchronize();

        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.0f; // ms -> s

        printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
               elapsed_time / repeat_times, 2.0 * 1e-9 * repeat_times * m * n * k / elapsed_time, m);
        fflush(stdout);
        copy_matrix(C_ref, C, m * n); // resync C with cuBLAS result before next size
    }

    free(A); free(B); free(C); free(C_ref);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
    cublasDestroy(handle);
    return 0;
}
