// Implementing the native kernel but using memory coalescing this time

// Dimensions of matrix A would be M * K
// Dimensooms of matrix B would be K * N 
// Therefore, dimensions of matric C would be M * N

// This kernel performs memory coalescing. A warp would execute threads with examples like (5, 8), (6, 8), (7, 8), (8, 8).
// In the above threads, the x coordinate is the col and the y coordinate is the row
// 
//  At one instant, say i = 7, all threads in the warp execute the same instruction.
// For threads mapped to output coordinates:
//
// (row, col) = (8,5), (8,6), (8,7), ...
//
// the A access is:
//
// A[8][7], A[8][7], A[8][7], ...
//
// so every thread accesses the same A element.
//
// The B access is:
//
// B[7][5], B[7][6], B[7][7], B[7][8], ...
//
// so the threads access consecutive elements from row 7 of B.
// Because B is stored in row-major order, those addresses are consecutive,
// so the B load is coalesced.


// The reason threads are organised like that is because the threadId = threadIdx.x + blockDim.x * threadIdx.y and wraps always put continous threadIds together.


__global__ void memory_coalesce(int M, int K, int N, float alpha, float beta, float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    float sum = 0; 

    if (row < M && col < N) {

        for (int i = 0; i < K; ++i) {
            // Linearise the coordinate to retrives values from A and B
            sum += A[row * K + i] * B[i * N + col]; 
        }

        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }

}



// Another way to write this kernel is down below

__global__ void memory_coalesce_alternative(int M, int N, int K, float alpha, float beta, float* A, float* B, float*C) {
    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);


    float sum = 0; 

    if (row < M && col < N) {

        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col]; 
        }

        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }

}

/*
This is how we would launch this kernel;

// gridDim stays the same
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
// make blockDim 1-dimensional, but don't change number of threads
dim3 blockDim(32 * 32);
sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

*/