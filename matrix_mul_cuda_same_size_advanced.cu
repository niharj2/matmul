// In matrix_mul_cuda_same_size, I had implemented a very simple version in which the kernel would deploy only 1 block, and 
// 1 block is able to contain only 1024 threads. This implementation uses blocks to handle mat mul of bigger sizes

__global__ void MatrixMultiplicationKernel(float* Md, float* Nd, float* Pd, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    float sum = 0;
    for (int i = 0; i < width; ++i) {
        float a = Md[row * width + i];
        float b = Nd[col + width * i];
        sum += a * b;
    }

    Pd[row * width + col] = sum; 
}

void MatrixMultiplication(float* M, float* N, float* P, int width, int tile_width) {
    float* Md; 
    float* Nd; 
    float* Pd; 

    int size = width * width * sizeof(float);
    // essentially Malloc does something like *&Md = GPU memory
    // &Md contains the address of Md in the CPU stack
    // *Md gives the value of the first element it points to. Chaning *Md to something else would just change the value of that memory address. It wouldn;t change what Md points to anyways. We need to change what Md points to now

    cudaMalloc((void**)&Md, size); 
    cudaMalloc((void**)&Nd, size); 
    cudaMalloc((void**)&Pd, size); 

    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice); 

    dim3 dimGrid(width / tile_width, width / tile_width); 
    dim3 dimBlock(tile_width, tile_width);

    MatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width); 


    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost); 


    cudaFree(Md); 
    cudaFree(Nd);
    cudaFree(Pd);
}


