// The idea is the same as 1D but instead of 1 thread just owning accum no of elements, 1 thread instead now has accum * accum elements
template <int BM, int BN, int BK, int accum>
__global__ void two_d_block_tiling_kernel(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    __shared__ float As[BM][BK]; 
    __shared__ float Bs[BK][BN]; 

    // block will be launched as dim block3(BM / accum, BN / accum) because now we are dealing with 2D block tiling

    int threadId = threadIdx.x + blockDim.x * threadIdx.y;
    int numberOfThreads = (BM * BN) / (accum * accum); 
    int numTiles = (K + BK - 1) / BK; 

    for (tile = 0; tile < numTiles; ++tile) {
        // Since the number of threads would decrease significantly, we can no longer have a config where the each thread would load only one element. 
        // Therefore, we would have for loops to load these elements. The number of elements we would load in one go would be BM * BK for As and BK * BN 
        int numAelements = BM * BK;
        int loading_A_phases = (numAelements + numberOfThreads - 1) / numberOfThreads;
        // AN IMPORTANT THING TO REMEMBER HERE IS THIS CONFIG WOULD WORK ONLY FOR THE CASE WHERE BLOCKDIM.X == BK; 
        for (phase = 0; phase < loading_A_phases; ++phase) {
            Arow = (phase * BK) + threadIdx.y
            Acol = threadIdx.x
            As[Arow][Acol] = A[(BM * blockIdx.y + Arow) * K + tile * BK + Acol];
        }

        int numBelements = BK * BN; 
        int loading_B_phases = (numBelements + numberOfThreads - 1) / numberOfThreads; 
        for (phase = 0; phase < loading_B_phases; ++phase) {
            Brow = threadIdx.y; 
            Bcol = (phase * BK) + threadIdx.x;
            Bs[Brow][Bcol] = B[(tile * BK + B_row) * N + (BN * blockIdx.x + Bcol)]
        }
    }

}