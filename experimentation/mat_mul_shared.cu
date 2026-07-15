"""
we have the global matrix, and in that we have blocks. For each block, we have a shared memory. 

The no of phases for each block execution would essentially be N / Title Width

Phases refer to the input, and the blocks refer to the output

Here's everything we covered:

---

## 1. The Three Levels: Total, Block, Phase

- **Total** = the entire problem (all elements of output Pd to compute)
- **Block** = which tile of the output this group of threads owns
- **Phase** = the sequential steps one block takes to finish its job

---

## 2. Blocks → divide the OUTPUT

- The output Pd is split into **TILE_WIDTH × TILE_WIDTH** tiles
- Each block owns one tile and is fully responsible for computing it
- All blocks run **in parallel** with each other

---

## 3. Phases → divide the INPUT

- The input (Md and Nd) is NOT pre-divided — it's just global memory
- Each block reads **different strips** of the input depending on which output tile it owns
- Because shared memory is limited, the input strip is eaten in **chunks (phases)** sequentially
- Number of phases = **N / TILE_WIDTH**

---

## 4. Within a Block

- Threads within a block run **in parallel** within each phase
- Each thread loads **exactly one element** into shared memory per phase
- This is why TILE_WIDTH controls both block size AND phase chunk size — they're locked together
- **`__syncthreads()`** ensures all threads finish loading before any thread starts computing

---

## 5. PValue

- Each thread has its own private **PValue** accumulator
- Each phase adds a partial dot product to it
- After all phases, PValue is the final answer → written to Pd

"""


__global__ void MatrixMultiplicationKernel(float* Md, float* Nd, float* Pd, int Width) {

    __shared__ float Mds[Title_Width][Title_Width]; 
    __shared__ float Nds[Title_Width][Title_Width]; 

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Remember the formula: block_id * block_size + thread; 
    // block ID would be global and the thread id would be local 
    // each block has TILE_WIDTH × TILE_WIDTH threads and the threads are numbered from 1 to title_width - 1
    
    int Row = Title_Width * by + ty; 
    int Col = Title_Width * bx + tx; 

    float Pval = 0; 

    // The outer loop - phases. 
    // m = which phase we're on, Runs N/TILE_WIDTH times
    for (int m = 0; m < Width / Title_Width; ++m) {

        // Loading into shared memory
        // If you look at it the access is still basically Row * Width + Col for 1D arrays
        // Essentially the m * Title_width will move it for every phase
        // Each thread would load every input in Md and Nd 
        // In CUDA, ty essentially means the row and the tx means the col 
        // for Md, the Col = m * Title_Width + tx
        // For Nd, the Row = m * Title_Width + ty 
        // m * TILE_WIDTH is what shifts to the next chunk each phase — this is the phase advancing through the input

        Mds[ty][tx] = Md[Row * Width + (m * Title_Width + tx)]; 
        Nds[ty][tx] = Nd[(m * Title_Width + ty)* Width + Col]; 

        // wait till all the threads of that particular phase load into their memory
        __syncthreads();

        // Performs the matrixmultiplication for that phase for that set
        for (int k = 0; k < Title_Width; ++k) {
            Pval += Mds[ty][k] * Nds[k][tx]; 
        }
         __syncthreads(); 
    }

    Pd[Row * Width + Col] = Pval;

}



void MatrixMultiplication(float* M, float* N, float* P, int Width, int Title_Width) {
    float* Md; 
    float* Nd; 
    float* Pd;

    int size = Width * Width * sizeof(float); 
    cudaMalloc((void**)&Md, size); 
    cudaMalloc((void**)&Nd, size); 
    cudaMalloc((void**)&Pd, size); 

    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice); 

    // here goes the matrix multiplication kernel 
    dim3 dimGrid(Width / Title_Width, Width / Title_Width);
    dim3 dimBlock(Title_Width, Title_Width);

    MatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width); 

    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

    cudaFree(Md); 
    cudaFree(Nd);
    cudaFree(Pd);
}