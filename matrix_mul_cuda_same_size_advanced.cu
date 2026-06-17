// In matrix_mul_cuda_same_size, I had implemented a very simple version in which the kernel would deploy only 1 block, and 
// 1 block is able to contain only 1024 threads. This implementation uses blocks to handle mat mul of bigger sizes

__global__ void MatrixMultiplicationKernel(float* Md, float* Nd, float* Pd, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    float sum = 0;
    for (int i = 0; i < width; ++i) {
        float a = Md[row * width + i];
        float b = Nd[col + width * i];
        sum += a + b
    }

    Pd[row * width][col] = sum; 
}


