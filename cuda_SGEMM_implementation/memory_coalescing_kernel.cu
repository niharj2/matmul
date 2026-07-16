// Implementing the native kernel but using memory coalescing this time

// Dimensions of matrix A would be M * K
// Dimensooms of matrix B would be K * N 
// Therefore, dimensions of matric C would be M * N


//  For execution, threads are organised into a group of 32, known as a wrap. A warp is then assigned to a warp scheduler, which is the physical core that executes the instructions
// A SM, has 4 wrap schedulers. The grouping into warps happens based on a consecutive threadId.

// Threads inside a block are assigned a linear threadId:
//
// threadId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
//
// A 3D block can be viewed as a stack of 2D grids.
//
// Within each 2D grid, threads are numbered in row-major order:
// x (columns) changes first, then y (rows).
//
// After all threads in one 2D slice have been numbered,
// numbering continues in the next z slice.
//
// This ordering groups neighboring x threads together,
// which matches row-major memory layout and helps achieve
// coalesced global memory accesses.


__global__ void memory_coalesce(int M, int K, int N, float alpha, float beta, float* A, float* B, float* C) {
    // setting up the row and col coords

    __shared__ float Ads[Title_Width][Title_Width]; 
    __shared__ float Bds[Title_Width][Title_Width];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // x represents cols and y represents rows
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    // making sure we don't go out of memory bounds

    if (row < M && col < N) {
        float sum = 0; 

        for (int m = 0; m < Width / Title_Width; m++) {
            Ads[ty][tx] = A[row * K + (m * Title_Width + tx)];
            Bds[ty][tx] = B[(m * Title_Width + ty) * N + col];
        }

        __syncthreads();


        for (int i = 0; i < K; ++i) {
            sum += Ads[ty][i] * Bds[i][tx];
        }

        __syncthreads();

        C[row * N + col] = sum;
    }
    
}