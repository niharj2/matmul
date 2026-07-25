# 1D Thread Tiling Notes

## Matrix Dimensions

For matrix multiplication:

- A: `M × K`
- B: `K × N`
- C: `M × N`

where

- `M` = number of rows of A (and C)
- `N` = number of columns of B (and C)
- `K` = reduction dimension (dot-product length)

---

# Block Tile Dimensions

Instead of computing the whole output matrix at once, each CUDA block computes a **tile**.

The tile dimensions are:

- `BM` = output rows computed by one block
- `BN` = output columns computed by one block

Therefore each block computes a tile of

```
BM × BN
```

For example,

```
BM = 64
BN = 64
```

means one block computes

```
64 × 64 = 4096
```

output elements.

---

# Phase Depth (BK)

The reduction dimension is not processed all at once.

Instead, it is divided into chunks.

```
BK = reduction depth processed per phase
```

Suppose

```
K = 1024
BK = 8
```

Then there are

```
1024 / 8 = 128 phases
```

During each phase, the block processes only

```
8
```

values along K.

---

# Shared Memory Tiles

The shared memory tiles are **NOT**

```
BM × K
```

They are

```
A tile = BM × BK
B tile = BK × BN
```

Example:

```
BM = 64
BN = 64
BK = 8
```

Shared memory stores

```
A tile = 64 × 8
B tile = 8 × 64
```

After every phase,

the next BK slice is loaded.

---

# General Shared Memory Declaration

Instead of

```cpp
__shared__ float As[TILE_WIDTH][TILE_WIDTH];
__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
```

the general version becomes

```cpp
__shared__ float As[BM][BK];
__shared__ float Bs[BK][BN];
```

---

# Why Old Kernels Used TILE_WIDTH × TILE_WIDTH

In the simple tiled kernel,

```
BM = BN = BK = TILE_WIDTH
```

For example,

```
TILE_WIDTH = 32
```

means

```
BM = 32
BN = 32
BK = 32
```

Therefore

```
As = 32 × 32
Bs = 32 × 32
```

This is why the simple implementation appears square.

---

# Number of Threads

Without thread tiling,

every thread computes

```
1 output
```

Therefore

```
Threads/block = BM × BN
```

Example

```
BM = 8
BN = 8

Threads = 64
```

---

# Thread Coarsening

Instead of computing one output,

each thread computes multiple outputs.

Suppose

```
TM = 4
```

Then

```
1 thread → 4 outputs
```

For an

```
8 × 8
```

output tile,

```
64 outputs
```

are still required,

so only

```
64 / 4 = 16
```

threads are needed.

---

# Two Possible Thread Mappings

There are two common mappings.

---

## 1. Horizontal Coarsening

One thread computes

```
C[row][col]
C[row][col+1]
C[row][col+2]
C[row][col+3]
```

Same row.

Different columns.

Each thread reuses

```
A[row][k]
```

because the same A value contributes to multiple columns.

---

## 2. Vertical Coarsening

One thread computes

```
C[row][col]
C[row+1][col]
C[row+2][col]
C[row+3][col]
```

Different rows.

Same column.

Each thread reuses

```
B[k][col]
```

because the same B value contributes to multiple rows.

---

# Which One Is Faster?

Both work.

The difference is what gets reused.

## Horizontal

Reuses

```
A
```

inside the thread.

## Vertical

Reuses

```
B
```

inside the thread.

For a straightforward scalar CUDA implementation,

vertical coarsening is usually preferred because adjacent threads naturally write consecutive columns, giving coalesced global-memory stores.

Horizontal coarsening can also perform well, especially when vectorized stores (`float4`, etc.) are used.

---

# Thread Mapping Depends On Your Design

CUDA does **not** force

```
threadIdx.x
```

to represent rows or columns.

You choose the mapping.

Examples:

### Mapping A

```
threadIdx.x → output column
threadIdx.y → output row group
```

### Mapping B

```
threadIdx.x → output row group
threadIdx.y → output column
```

Both are correct.

The indexing inside the kernel must simply match the mapping you choose.

---

# Important Distinction

Do **NOT** confuse

```
CUDA thread block dimensions
```

with

```
output tile dimensions.
```

Example:

Output tile

```
64 × 64 outputs
```

does **not** necessarily mean

```
64 × 64 CUDA threads.
```

With thread coarsening,

you might instead launch only

```
16 × 16 threads
```

while every thread computes multiple outputs.

---

# Summary

Matrix dimensions

```
A = M × K
B = K × N
C = M × N
```

Block computes

```
BM × BN outputs
```

Each phase processes

```
BK
```

values of K.

Shared memory stores

```
As = BM × BK
Bs = BK × BN
```

Without thread tiling

```
1 thread = 1 output
```

With thread tiling

```
1 thread = TM outputs
```

Possible mappings

```
Horizontal:
same row
different columns

Vertical:
different rows
same column
```

Horizontal reuses A.

Vertical reuses B.