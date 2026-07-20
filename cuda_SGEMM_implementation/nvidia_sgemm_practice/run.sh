#!/bin/bash
# Run all kernels (0=cublas, 1=native, 2=coalesce, 3=coalesce_alt, 4=shared)
# and save each run's output for plot.py to parse.
set -e
cd "$(dirname "$0")"
rm -f ./test/test_kernel*
for i in 0 1 2 3 4; do
    echo -n "kernel ${i}... "
    ./sgemm ${i} > ./test/test_kernel_${i}.txt
    echo "done"
done
