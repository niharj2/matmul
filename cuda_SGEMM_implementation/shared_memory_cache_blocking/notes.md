Next to the large global memory, a GPU has a much smaller region of memory that is physically located on the chip, called shared memory (SMEM)

 Physically, there’s one shared memory per SM. All threads can access this shared memory within a particular block.

 The amount of SMEM is configurable, by trading off a larger shared memory for a smaller L1 cache.

As the shared memory is located on-chip, it has a much lower latency and higher bandwidth than global memory


# Execution Model of Tiled Matrix Multiplication

## Kernel Launch

```cpp
kernel<<<grid, block>>>();
```

CUDA creates **one thread for every `(blockIdx, threadIdx)` combination**.

Every thread executes the **same kernel code**, but each thread has different values for:

- `threadIdx`
- `blockIdx`
- `row`
- `col`
- `sum`

These variables are private to each thread (stored in registers).

---

## Local vs Global Indices

`threadIdx` is **local to a block**.

For every block:

```text
threadIdx.x = 0 ... blockDim.x - 1
threadIdx.y = 0 ... blockDim.y - 1
```

These values **reset for every block**.

Global coordinates are computed as

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

These map the thread to one output element:

```text
Thread(row, col) ---> C[row][col]
```

---

# Blocks

Each block computes **one output tile**.

Example (`TILE_WIDTH = 2`):

```text
Block (0,0)

+----+----+
|C00 |C01 |
+----+----+
|C10 |C11 |
+----+----+
```

Another block computes another tile.

Blocks are completely independent.

---

# Shared Memory

```cpp
__shared__ float As[TILE_WIDTH][TILE_WIDTH];
__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
```

These arrays belong to **one block only**.

Every block gets its **own private copy** of

- `As`
- `Bs`

Shared memory is **not global**.

---

# Shared Memory Lives on the SM

Each Streaming Multiprocessor (SM) contains its own

- Shared Memory
- Registers
- Warp Schedulers

Example:

```text
GPU

SM0
 ├── Shared Memory
 ├── Block0
 └── Block1

SM1
 ├── Shared Memory
 ├── Block2
 └── Block3
```

There is **no single shared memory for the whole GPU**.

---

# Not All Blocks Execute Together

Launching

```text
1000 blocks
```

does **NOT** mean 1000 blocks run simultaneously.

Only as many blocks as fit on the available SMs execute.

Remaining blocks wait until an SM becomes available.

Therefore:

- Block A may be on Tile 3
- Block B may be on Tile 1
- Block C may not have started yet

There is **no global synchronization between blocks**.

---

# Tile Phases

Inside one block:

```cpp
for (tile = 0; tile < numTiles; tile++)
```

Every thread loads

- one element from A
- one element from B

into that block's shared memory.

```cpp
As[ty][tx] = ...
Bs[ty][tx] = ...
```

After all threads load their elements, the block computes using the tile.

Then the shared memory is reused for the next tile.

---

# __syncthreads()

`__syncthreads()` synchronizes **only threads in the same block**.

Example:

```text
Block A

T0
T1
T2
T3
```

These four threads wait for each other.

They do **NOT** wait for

```text
Block B

T4
T5
T6
T7
```

Different blocks never synchronize inside a kernel.

---

# Why Two __syncthreads()?

### First barrier

```cpp
Load tile

__syncthreads()
```

Ensures **every thread has finished loading** before any thread starts computing.

---

### Second barrier

```cpp
Compute

__syncthreads()
```

Ensures **every thread has finished using the tile** before any thread overwrites shared memory with the next tile.

---

# Mental Model

Think of each block as an independent team.

Each team has:

- its own threads
- its own shared memory
- its own synchronization

The GPU schedules many such teams across the available SMs.

Each team progresses through tile phases independently and never shares its `As` or `Bs` arrays with other blocks.


There are three main limits to keeping more active blocks loaded on an SM: register count, warp count and SMEM capacity