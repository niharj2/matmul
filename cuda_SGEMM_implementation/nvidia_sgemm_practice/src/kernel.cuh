#pragma once

// Tunables the included kernels depend on.
#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#ifndef BLOCKSIZE
#define BLOCKSIZE 32
#endif
#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// Pull in the real kernels from the project (single source of truth —
// nothing here reimplements them).
#include "../../native/native_kernel.cu"
#include "../../memory_coalescing/memory_coalescing_kernel.cu"
#include "../../shared_memory_cache_blocking/shared_memory_kernel.cu"
