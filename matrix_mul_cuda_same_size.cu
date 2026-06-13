// This is a VERY SIMPLE implementation of matrix multiplication of 2 matrices of same size N by N 

/* 

MatrixMultiplication(), is responsible for allocating device memory, performing data transfers, and activating the kernel that performs the actual matrix multiplication.
This program would have mainly 3 parts. 

Part one would involve storing the input matrices and the product matrix from the host (CPU) into the device memory (GPU).
This would be stored onto the global memory for now. So essentially we allocate device memory and copy essential elements onto it. 

Part 2 would involve invoking a kernel that would launch the parallel execution of matrix multiplication. 

Part 3 would involve freeing up space in the device (GPU) and sending back data to the host (CPU)


Essential functions:
cudaMalloc allocates memory on the GPU and stores the address of that allocation into Md
cudaFree() - frees object from device global memory
*/

// Implementing Part 1 

// &md is the address of where md is stored on the cpu, and md gives the address of the stored elements inside it


// the kernel for matrix multiplication

__global__ void MatrixMultiplicationKernel(float* Md, float* Nd, float* Pd, int width) {
        int tx = threadIdx.x; 
        int ty = threadIdx.y; 
        
        float sum = 0;
        for (int i = 0; i < width; ++i) {
            float a = Md[ty * width + i];
            float b = Nd[i * width + tx];
            sum += a * b;
        }
        Pd[ty * width + tx] = sum; 
    }


void MatrixMultiplication(float* M, float* N, float* P, int width) {
    float* Md; 
    float* Nd; 
    float* Pd;
    
    size_t size = width * width * sizeof(float);

    // Allocate memory in the device
    cudaMalloc((void**) &Md, size); 
    cudaMalloc((void**) &Nd, size); 
    cudaMalloc((void**) &Pd, size);

    //Copy data from the host to the device
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(width, width); 
    dim3 dimGrid(1, 1); 

    MatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width); 

    // copy the product from device to host 
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost); 


    // free memory on device 
    cudaFree(Md); 
    cudaFree(Nd);
    cudaFree(Pd);
}


// the kernel for matrix multiplication
__global__ void MatrixMultiplication(float* Md, float* Nd, float* Pd, int width) {
        int tx = threadIdx.x; 
        int ty = threadIdx.y; 
        
        float sum = 0;
        for (int i = 0; i < width; ++i) {
            float a = Md[tx * width + i];
            float b = Nd[i * width + ty];
            sum += a * b;
        }
        Pd[tx * width + ty] = sum; 
    }




