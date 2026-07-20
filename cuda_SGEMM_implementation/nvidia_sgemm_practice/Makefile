# H200 is sm_90.
NVCC      ?= nvcc
ARCH      ?= sm_90
TILE_WIDTH ?= 32
BLOCKSIZE  ?= 32
NVCCFLAGS  = -O3 -arch=$(ARCH) -Isrc -DTILE_WIDTH=$(TILE_WIDTH) -DBLOCKSIZE=$(BLOCKSIZE) -lcublas

sgemm: sgemm.cu src/utils.cu src/utils.cuh src/kernel.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ sgemm.cu src/utils.cu

clean:
	rm -f sgemm

.PHONY: clean
