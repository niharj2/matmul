// Adapted from wangzyon/NVIDIA_SGEMM_PRACTICE's sgemm.cu, wired to test the
// kernels in cuda_SGEMM_implementation instead of that repo's mysgemm_v*.
//
// kernel_num: 0=cuBLAS, 1=native, 2=coalesce, 3=coalesce_alt, 4=shared
//
// Usage:
//   ./sgemm <kernel_num>            square sweep M=N=K = 256, 512, ..., 6144
//   ./sgemm <kernel_num> M N K       single run at the given (possibly
//                                    rectangular) shape
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

static void bench_one(int kernel_num, int m, int n, int k, float alpha, float beta,
                      cublasHandle_t handle, int repeat_times) {
    printf("m=%d n=%d k=%d\n", m, n, k);

    float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL;
    float *dA = NULL, *dB = NULL, *dC = NULL, *dC_ref = NULL;

    A = (float *) malloc(sizeof(float) * m * k);
    B = (float *) malloc(sizeof(float) * k * n);
    C = (float *) malloc(sizeof(float) * m * n);
    C_ref = (float *) malloc(sizeof(float) * m * n);

    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);
    randomize_matrix(C, m * n);
    copy_matrix(C, C_ref, m * n);

    cudaCheck(cudaMalloc((void **) &dA, sizeof(float) * m * k));
    cudaCheck(cudaMalloc((void **) &dB, sizeof(float) * k * n));
    cudaCheck(cudaMalloc((void **) &dC, sizeof(float) * m * n));
    cudaCheck(cudaMalloc((void **) &dC_ref, sizeof(float) * m * n));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * m * n, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C_ref, sizeof(float) * m * n, cudaMemcpyHostToDevice));

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
    } else {
        // cuBLAS itself needs its own warm-up call (algorithm/plan selection
        // on the first call can otherwise leak into the timed average).
        test_kernel(0, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaDeviceSynchronize();

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
        test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.0f; // ms -> s

    // Report a single "size" that keeps plot.py's parser working for square
    // sweeps; for rectangular shapes this is just m (the M dimension).
    printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
           elapsed_time / repeat_times, 2.0 * 1e-9 * repeat_times * (double) m * n * k / elapsed_time, m);
    fflush(stdout);

    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    free(A); free(B); free(C); free(C_ref);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 5) {
        printf("Usage:\n"
               "  %s <kernel_num>            square sweep M=N=K=256..6144\n"
               "  %s <kernel_num> M N K      single run at that shape\n"
               "kernel_num: 0=cublas, 1=native, 2=coalesce, 3=coalesce_alt, 4=shared\n",
               argv[0], argv[0]);
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

    float alpha = 1.0f, beta = 0.0f; // C = alpha*A*B + beta*C
    int repeat_times = 10;

    if (argc == 5) {
        int m = atoi(argv[2]);
        int n = atoi(argv[3]);
        int k = atoi(argv[4]);
        bench_one(kernel_num, m, n, k, alpha, beta, handle, repeat_times);
    } else {
        int size_len = 24;
        int SIZE[size_len];
        for (int i = 0; i < size_len; i++)
            SIZE[i] = 256 * (i + 1);
        printf("max_size=%d\n", SIZE[size_len - 1]);

        for (int i = 0; i < size_len; i++)
            bench_one(kernel_num, SIZE[i], SIZE[i], SIZE[i], alpha, beta, handle, repeat_times);
    }

    cublasDestroy(handle);
    return 0;
}
