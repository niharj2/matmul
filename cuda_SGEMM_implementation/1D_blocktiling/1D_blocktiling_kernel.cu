__global__ void 1D_blocktiling_kernel(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta, int accum, int BM, int BN, int BK) {
    __shared__ float Ad[BM][BK]; 
    __shared__ float Bd[BK][BN]; 

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int bx = blockIdx.x; 
    int by = blockIdx.y; 

    float sum[accum];
    int row = 
    int numTiles = (K + BK - 1) / BK; // Essentially gives us the number of phases out there

    for (tile = 0; tile < numTiles; ++tile) {
        // Each thread now handles accum no of output elements. The no of elements each thread loads is not necessarily the same as the no of output elements it handles
        // The no of threads per block will be (BM * BN) / TM
        A_col = tile * BK + tx;
        Ad[ty][tx] = A[row * K + A_col];

        B_row = (tile * BK + ty) * K;
        Bd[ty][tx] = B[B_row + col];

        __syncthreads(); 

        for (element = 0; element < accum; ++element) {
            float B_cache = Bd[]
        }

    }

}