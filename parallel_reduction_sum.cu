// Kernel to calculate the sum of 1D array parallely 

__global__ void ParallelSumReduction(float* Md, int Width) {
    // extern helps us set the Mds size at runtime instead of compile time
    extern __shared__ float Mds[];
    int tx = threadIdx.x;

    // Loads it into the shared memory. A very simple implementation here. 
    Mds[tx] = Md[tx]; 
    __syncthreads();

    for (int stride = 1; stride < Width; stride *= 2) {
        if (tx % (2 * stride) == 0) {
            Mds[tx] += Mds[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        Md[tx] = Mds[tx]; 
    } 
}