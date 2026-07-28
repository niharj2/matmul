// Do not get confused between the global index and linearisation.
// By global index, I mean the global index of the thread.
// Because CUDA divides the grid into different blocks, thread indices
// are refreshed inside every block.
//
// By refreshed, we mean that threads are numbered from
// 0 to blockDim - 1 inside every block.
//
// Linearisation is the process of converting a 2D or 3D coordinate
// into one memory index.

__global__ void shared_memory_kernel_same_size(int N, float* A, float* B, float* C, float alpha,float beta) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global thread coordinates mapped to C.
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    float sum = 0;

    // We want every thread to load one element PER PHASE.
    for (int tile = 0; tile < numTiles; ++tile) {

        // Column in A that this thread loads during this phase.
        int A_col = tile * TILE_WIDTH + tx;

        // Row in B that this thread loads during this phase.
        int B_row = tile * TILE_WIDTH + ty;

        // Check whether the A element is valid.
        if (row < N && A_col < N) {
            As[ty][tx] = A[row * N + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Check whether the B element is valid.
        if (B_row < N && col < N) {
            Bs[ty][tx] = B[B_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Some threads in the final blocks may map outside C.
    if (row < N && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Matrix A dimension = M * K
// Matrix B dimension = K * N 
// Product dimension = M * N

__global__ void shared_memory_kernel(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int bx = blockIdx.x; 
    int by = blockIdx.y; 

    int row = by * blockDim.y + ty; 
    int col = bx * blockDim.x + tx;

    float sum = 0;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH; 

    for (int tile = 0; tile < numTiles; ++tile) {
        // Every thread is reponsible for loading one element per phase
        int A_col = tile * TILE_WIDTH + tx;

        if (row < M && A_col < K) {
            As[ty][tx] = A[row * K + A_col];
        } else {
            As[ty][tx] = 0;
        }
        
        int B_row = tile * TILE_WIDTH + ty;

        if (B_row < K && col < N) {
            Bs[ty][tx] = B[B_row * N + col]; 
        } else {
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }

}