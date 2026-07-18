# Build the SGEMM harness. H200 is sm_90.
NVCC      ?= nvcc
ARCH      ?= sm_90
TILE_WIDTH ?= 32
BLOCKSIZE  ?= 32
NVCCFLAGS  = -O3 -arch=$(ARCH) -DTILE_WIDTH=$(TILE_WIDTH) -DBLOCKSIZE=$(BLOCKSIZE) -lcublas

sgemm: sgemm_harness.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Convenience: tile-size sweep builds (experiment #6).
sweep:
	$(NVCC) $(NVCCFLAGS) -DTILE_WIDTH=8  -o sgemm_tw8  sgemm_harness.cu
	$(NVCC) $(NVCCFLAGS) -DTILE_WIDTH=16 -o sgemm_tw16 sgemm_harness.cu
	$(NVCC) $(NVCCFLAGS) -DTILE_WIDTH=32 -o sgemm_tw32 sgemm_harness.cu

clean:
	rm -f sgemm sgemm_tw8 sgemm_tw16 sgemm_tw32

.PHONY: sweep clean
