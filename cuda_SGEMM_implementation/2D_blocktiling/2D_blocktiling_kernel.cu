#include "two_d_block_tiling.hpp"
// The idea is the same as 1D but instead of 1 thread just owning accum no of elements, 1 thread instead now has accum * accum elements
template <int BM, int BN, int BK, int accum>
__global__ void two_d_block_tiling_kernel(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    __shared__ float As[BM][BK]; 
    __shared__ float Bs[BK][BN]; 

    // block will be launched as dim block3(BM / accum, BN / accum) because now we are dealing with 2D block tiling

    int numberOfThreads = (BM * BN) / (accum * accum); 
    int numTiles = (K + BK - 1) / BK;

    float sum_accum[accum][accum];
    for (int r = 0; r < accum; ++r)
        for (int c = 0; c < accum; ++c)
            sum_accum[r][c] = 0.0f;

    for (int tile = 0; tile < numTiles; ++tile) {
        // Since the number of threads would decrease significantly, we can no longer have a config where the each thread would load only one element. 
        // Therefore, we would have for loops to load these elements. The number of elements we would load in one go would be BM * BK for As and BK * BN 
        int numAelements = BM * BK;
        int loading_A_phases = (numAelements + numberOfThreads - 1) / numberOfThreads;
        // AN IMPORTANT THING TO REMEMBER HERE IS THIS CONFIG WOULD WORK ONLY FOR THE CASE WHERE BLOCKDIM.X == BK; 
        for (int phase = 0; phase < loading_A_phases; ++phase) {
            int Arow = (phase * BK) + threadIdx.y;
            int Acol = threadIdx.x;
            As[Arow][Acol] = A[(BM * blockIdx.y + Arow) * K + tile * BK + Acol];
        }

        int numBelements = BK * BN; 
        int loading_B_phases = (numBelements + numberOfThreads - 1) / numberOfThreads; 
        for (int phase = 0; phase < loading_B_phases; ++phase) {
            int Brow = threadIdx.y;
            int Bcol = (phase * BK) + threadIdx.x;
            Bs[Brow][Bcol] = B[(tile * BK + Brow) * N + (BN * blockIdx.x + Bcol)];
        }

        __syncthreads();

        // computing the values
        for (int i = 0; i < BK; ++i) {
            float B_reg[accum]; 
            for (int col_b_index = 0; col_b_index < accum; ++col_b_index) {
                B_reg[col_b_index] = Bs[i][threadIdx.x * accum + col_b_index];
            }

            for (int row_a_index = 0; row_a_index < accum; ++row_a_index) {
                float A_cached_value = As[threadIdx.y * accum + row_a_index][i];
                for (int j = 0; j < accum; ++j) {
                    sum_accum[row_a_index][j] += A_cached_value * B_reg[j];
                }
            }
        }
        __syncthreads();
    }

    int globalColBase = blockIdx.x * BN + threadIdx.x * accum;
    int globalRowBase = blockIdx.y * BM + threadIdx.y * accum;

    for (int row_index = 0; row_index < accum; ++row_index) {
        for (int col_index = 0; col_index < accum; ++col_index) {
            int globalRow = globalRowBase + row_index;
            int globalCol = globalColBase + col_index;
            C[globalRow * N + globalCol] = alpha * sum_accum[row_index][col_index] + beta * C[globalRow * N + globalCol];
        }
    }
}


void launch_2d(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
    dim3 block(128/8, 128/8);                    // (16,16) = 256 threads
    dim3 grid((N+127)/128, (M+127)/128);
    two_d_block_tiling_kernel<128,128,16,8><<<grid, block>>>(M, N, K, A, B, C, alpha, beta);
}
