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