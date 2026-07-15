// Implementing a native CUDA SGEMM kernel. SGEMM performs C = αAB+βC at single (=32b) precision.

// Dimensions of matrix A would be M * K
// Dimensooms of matrix B would be K * N 
// Therefore, dimensions of matric C would be M * N

__global__ void native_sgemm(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x; 

    // makes sure we aren't going out of bounds

    if (x < M && y < N) {
        float sum = 0; 
        for (int i = 0; i < K; ++i) {
            sum += A[x * K + i] * B[i * N + y];
        }
        
        // Using N instead of K, because the stride becomes N since that's the updated dimension of the matrix
        C[x * N + y] = alpha * sum + beta * C[x * N + y];
    }
}