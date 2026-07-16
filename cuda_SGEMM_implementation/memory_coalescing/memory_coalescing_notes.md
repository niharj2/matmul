````md
# CUDA SGEMM Notes

## Native SGEMM

### Thread Mapping

Each CUDA thread computes exactly **one element** of the output matrix `C`.

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

The thread `(row, col)` computes:

```text
C[row][col]
```

using the dot product of:

- Row `row` of `A`
- Column `col` of `B`

```text
C[row][col] = Σ A[row][i] × B[i][col]
```

---

### Boundary Check

```cpp
if (row < M && col < N)
```

CUDA always launches complete thread blocks.

If `M` or `N` is not a multiple of the block size, extra threads are created in the last block. These threads would otherwise access memory outside the matrix bounds, so they are ignored using the boundary check.

---

## Tiled SGEMM

### Shared Memory

Shared memory is used to cache small tiles of `A` and `B` inside each thread block.

```cpp
__shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
__shared__ float Bds[TILE_WIDTH][TILE_WIDTH];
```

Each thread loads exactly **one element** from `A` and **one element** from `B`:

```cpp
Ads[ty][tx] = ...
Bds[ty][tx] = ...
```

The entire thread block cooperatively fills both shared-memory tiles.

After

```cpp
__syncthreads();
```

every thread can access **every element** inside both shared-memory tiles.

---

### Phases

The dot product is computed one tile at a time along the shared dimension `K`.

Example:

```text
K = 64
TILE_WIDTH = 32
```

The computation is split into:

```
Phase 0:
K indices 0 → 31

Phase 1:
K indices 32 → 63
```

Number of phases:

```cpp
(K + TILE_WIDTH - 1) / TILE_WIDTH
```

Each phase follows the same sequence:

1. Load one tile of `A` and `B` into shared memory.
2. Synchronize all threads.
3. Compute the partial dot product using that tile.
4. Synchronize again.
5. Move to the next phase.

---

## Why Shared Memory Improves Performance

Shared memory **does not reduce memory usage**.

Instead, it reduces **global memory traffic**.

Without shared memory:

- Every thread repeatedly loads the same values from global memory.

With shared memory:

- Each value is loaded **once** from global memory.
- The entire thread block reuses that value many times from fast shared memory.

This significantly reduces expensive DRAM accesses.

---

# Memory Coalescing

## The Biggest Realization

I initially thought about **one thread executing over multiple loop iterations**.

That is **not** how memory coalescing is determined.

Memory coalescing is determined by:

> **What addresses are all 32 threads in a warp accessing during the same instruction?**

The memory system looks at one instruction across an entire warp—not one thread across time.

---

## Example

Suppose one warp computes:

```text
C[5][0]
C[5][1]
...
C[5][31]
```

Now consider a single loop iteration:

```cpp
i = 7;
```

Every thread executes simultaneously:

```cpp
A[row][i]
B[i][col]
```

### A accesses

Every thread accesses:

```text
A[5][7]
A[5][7]
A[5][7]
...
A[5][7]
```

All threads request the **same element** of `A`.

---

### B accesses

Every thread accesses:

```text
B[7][0]
B[7][1]
B[7][2]
...
B[7][31]
```

These are consecutive elements of the same row of `B`.

Therefore, the memory requests are **coalesced**.

---

## Final Mental Model

The key insight that finally made memory coalescing click:

I was following **one thread** through the loop.

CUDA does **not** evaluate memory coalescing that way.

Instead, think about one instant in execution.

At a particular iteration (for example, `i = 7`), **every thread in the warp is executing the exact same instruction simultaneously**.

So instead of asking:

> **What memory locations does my thread access over time?**

always ask:

> **At this exact instruction, what memory addresses are all 32 threads in the warp accessing?**

That single question determines whether the access is coalesced or not.
````


## Warp Execution

A warp consists of **32 threads** that execute the **same instruction at the same time**.

For a block launched as:

```cpp
dim3 block(32, 32);
```

the block contains:

```text
32 × 32 = 1024 threads
```

which means:

```text
1024 / 32 = 32 warps
```

Warps are formed by varying `threadIdx.x` first.

For example:

```text
Warp 0
(0,0) (1,0) (2,0) ... (31,0)

Warp 1
(0,1) (1,1) (2,1) ... (31,1)

...

Warp 5
(0,5) (1,5) (2,5) ... (31,5)
```

Each warp executes every instruction in lockstep.

Suppose Warp 5 is executing the loop iteration:

```cpp
i = 7;
```

Every thread in that warp executes the instruction

```cpp
A[row][i]
B[i][col]
```

**at exactly the same time.**

The accesses become:

```text
A

Thread (0,5)  -> A[5][7]
Thread (1,5)  -> A[5][7]
Thread (2,5)  -> A[5][7]
...
Thread (31,5) -> A[5][7]
```

All threads request the **same element** from `A`.

For `B`:

```text
Thread (0,5)  -> B[7][0]
Thread (1,5)  -> B[7][1]
Thread (2,5)  -> B[7][2]
...
Thread (31,5) -> B[7][31]
```

All threads request consecutive elements from the same row of `B`.

This is why the `B` access is coalesced.

> **Important**
>
> One thread accesses:
>
> ```
> A[5][0], A[5][1], ..., A[5][31]
> ```
>
> which is contiguous.
>
> However, **memory coalescing is NOT determined by one thread.**
>
> It is determined by what **all 32 threads in a warp** access during **one instruction**.







## Row-Major Memory Layout

Matrices in CUDA are typically stored in **row-major order**.

For a matrix `A` with dimensions `M × K`:

```cpp
A[row * K + col]
```

The row determines the starting location in memory, while the column determines the offset within that row.

Example (`K = 32`):

```text
A[5][0] -> 160
A[5][1] -> 161
A[5][2] -> 162
...
A[5][31] -> 191
```

All elements of the same row are stored consecutively in memory.

---

Moving **across columns**:

```text
A[5][3]
A[5][4]
A[5][5]
```

produces consecutive addresses:

```text
5*K + 3
5*K + 4
5*K + 5
```

---

Moving **down rows**:

```text
A[5][3]
A[6][3]
A[7][3]
```

produces addresses:

```text
5*K + 3
6*K + 3
7*K + 3
```

which are separated by `K` elements and are therefore **not consecutive**.

---

## Thread Mapping Determines Coalescing

The memory access expression alone does **not** determine whether accesses are coalesced.

For example, these two expressions are identical:

```cpp
A[row * K + i]
```

```cpp
A[x * K + i]
```

The difference is **how `row` (or `x`) is computed**.

### My Kernel

```cpp
row = threadIdx.y;
col = threadIdx.x;
```

Across a warp:

```text
row = constant
col = changing
```

At a fixed loop iteration:

```text
A[row][i]
A[row][i]
A[row][i]
...
```

All threads access the same element of `A`.

For `B`:

```text
B[i][0]
B[i][1]
...
B[i][31]
```

which are consecutive and therefore coalesced.

---

### Siboehm's Naive Kernel

```cpp
row = threadIdx.x;
col = threadIdx.y;
```

Across a warp:

```text
row = changing
col = constant
```

At a fixed loop iteration:

```text
A[0][i]
A[1][i]
A[2][i]
...
```

These accesses move down a column of `A`.

Since matrices are stored in row-major order, those addresses are separated by `K` elements and therefore are **not coalesced**.

---

## Final Intuition

Consecutive memory in a row-major matrix is achieved by changing the **column index**, not the row index.

Whether a warp accesses consecutive columns or consecutive rows depends entirely on how the programmer maps:

```cpp
threadIdx.x
threadIdx.y
```

to

```text
row
col
```

Changing only this mapping can completely change the memory coalescing behavior of the kernel, even if the matrix multiplication equation itself remains identical.



## Warp Formation

CUDA assigns every thread a **linear thread ID**.

For a 2D thread block:

```cpp
linearThreadId = threadIdx.x + blockDim.x * threadIdx.y;
```

Since `threadIdx.x` changes first, threads are ordered as:

```text
(0,0), (1,0), (2,0), ..., (31,0),
(0,1), (1,1), ...
```

A warp consists of **32 consecutive linear thread IDs**.

For a `32 × 32` block:

```text
Warp 0:
(0,0) ... (31,0)

Warp 1:
(0,1) ... (31,1)

Warp 2:
(0,2) ... (31,2)

...
```

Warps are therefore formed **horizontally** across `threadIdx.x`, **not vertically** across `threadIdx.y`.

---

## Why This Matters

Memory coalescing is evaluated **within a warp**.

Therefore, the memory accesses that matter are those made by threads like:

```text
(0,5), (1,5), (2,5), ..., (31,5)
```

and **not** by threads like:

```text
(5,0), (5,1), (5,2), ...
```

Those belong to different warps.

---

## Key Takeaway

Whether memory accesses are coalesced depends on:

1. How CUDA forms warps (consecutive linear thread IDs).
2. How `threadIdx.x` and `threadIdx.y` are mapped to matrix rows and columns.

Changing only the thread-to-matrix mapping can completely change the memory access pattern, even though the matrix multiplication equation remains identical.