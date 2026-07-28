// BM/BN/BK/accum are template params (not runtime args) because __shared__
// arrays and sum_accum[] must be sized at compile time.
template <int BM, int BN, int BK, int accum>
__global__ void one_d_block_tiling_kernel(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {

    //Global vs. local coordinates determines which element you load from global memory.
    //Number of threads vs. number of tile elements determines whether each thread needs to load multiple elements.


    // The main idea here is that each thread handles multiple outputs. The no of outputs that each thread calculates is given by the parameter accum
    // Matrix A is of size M * K. Matrix B is of size K * N
    // Each block is of size BM * BN. Each tile is of the size BM * BK for matrix A and BK * BN for matrix B. BK is essentially the parallel phase sum reduction we perform. 

    /*
    In this first kernel implementation, I am assuming that each thread will load only one element. An example of this is illustrated below
    
    Each block = 8 * 8, and each thread handles 4 elements 
    Therefore, no of threads would be (8 * 8) / 4 = 16
    Lets say that the phase value or BK value is 2. In that case, shared memory A is 8 * 2, and shared memory B is 2 * 8. In this case, each thread would be responsible for loading one element
    
    
    There are some important choices to be made here in terms of the mapping. 

    ## 1. Horizontal Coarsening
    One thread computes

    ```
    C[row][col]
    C[row][col+1]
    C[row][col+2]
    C[row][col+3]

    ```
    This means same row and diff col


    # Vertical Coarsening

    One thread computes

    ```
    C[row][col]
    C[row + 1][col]
    C[row + 2][col]
    C[row + 3][col]
    ```

    Same col, diff row. 
    In this first part of implementation, I'm gonna pick Vertical Coarsening. I think this would be faster because we would end up preserving coalesced global-memory accesses

    Since we are performing vertical coarsening, the kernel would be launched as follows: 
    dim3 block (BN, BM / accum)
    x would range from 0 to BN - 1. Complying with the standard CUDA practise, x would represent to be col
    y would range from 0 to (BM / accum) - 1. y would represent to be row
    */

    __shared__ float As[BM][BK]; 
    __shared__ float Bs[BK][BN]; 

    // In previous kernels I have always used the 2D form, where we have an x and y. However, as referring to various blog posts and best practices, I have found that this is easier to manupilate data loading 
    // The best practise here would be to linearise the threadIDs
    // logically the formula would be threadIdx.x + blockDim.x * threadIdx.y + blockDim.y * blockDim.x * threadIdx.z (The reason this is the formula is because we are flatening by row by row)
    // The threads are numbered from x - 0 to BN - 1 and y - 0 to BM / TM

    int threadId = threadIdx.x + blockDim.x * threadIdx.y; 

    float sum_accum[accum];
    for (int index = 0; index < accum; ++index) sum_accum[index] = 0.0f;
    int numTiles = (K + BK - 1) / BK;
    for (int tile = 0; tile < numTiles; ++tile){
        int Arow = threadId / BK;
        int Acol = threadId % BK;

        As[Arow][Acol] = A[(blockIdx.y * BM + Arow) * K + tile * BK + Acol];
        
        int Brow = threadId / BN;
        int Bcol = threadId % BN;

        Bs[Brow][Bcol] = B[(tile * BK + Brow) * N + blockIdx.x * BN + Bcol];
        __syncthreads(); 

        // We would now need 2 outer loops. One loop would refer to the output element we are calculating. For example, if each thread calculates 4 elements then the outper loop would be 0, 1, 2, 3

        for (int i = 0; i < BK; ++i) {
            // Since we will be using the same B value for all rows each thread is reponsible for calculating, we will cache that value
            // THIS IS THE WHOLE POINT
            float B_value = Bs[i][threadIdx.x];
            for (int index = 0; index < accum; ++index) {
                sum_accum[index] += As[threadIdx.y * accum + index][i] * B_value;
            }   
        }
        __syncthreads();

        // The above code was block focused. Notice that the threads were also focused on the blocks. In previous implementations of the kernel I always used global mapping of the rows and cols and thats why we never needed this piece of code until now. Do not TRIP on this
    }

    int globalCol = blockIdx.x * BN + threadIdx.x;
    int globalRowBase = blockIdx.y * BM + threadIdx.y * accum;

    for (int index = 0; index < accum; ++index) {
        int globalRow = globalRowBase + index;
        C[globalRow * N + globalCol] = alpha * sum_accum[index] + beta * C[globalRow * N + globalCol];
    }
}