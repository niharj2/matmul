#ifndef two_d_block_tiling
#define two_d_block_tiling

#pragma once

#include <cuda_runtime.h>


// The kernel itself is a template, so it cannot be declared here and defined
// elsewhere -- the definition must be visible where it is instantiated. This
// header exposes the plain host wrapper instead; the template is instantiated
// inside 2D_blocktiling_kernel.cu, which is where the tile config lives.
void launch_2d(
    int M,
    int N,
    int K,
    float* A,
    float* B,
    float* C,
    float alpha,
    float beta
);

#endif
