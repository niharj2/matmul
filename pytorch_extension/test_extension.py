"""
Demo / test for the `nihar_mat_mul` PyTorch extension.

`nihar_mat_mul.mat_mul(a, b)` runs a custom 2D block-tiling CUDA SGEMM kernel
and returns a @ b. This script:
  1. checks correctness against torch.matmul (cuBLAS) at several shapes,
  2. verifies the input-validation guards raise clean errors,
  3. does a quick speed comparison vs torch.matmul.

Build the extension first (from this directory):
    CUDA_HOME=/home/niharj2/.conda/envs/cuda-dev \
      /home/niharj2/.conda/envs/cuda-dev/bin/python setup.py build_ext --inplace

Run:
    /home/niharj2/.conda/envs/cuda-dev/bin/python test_extension.py

Note: the kernel is instantiated as <BM=128, BN=128, BK=16, TM=8> with no
boundary checks, so it requires M % 128 == 0, N % 128 == 0, K % 16 == 0.
"""

import torch
import nihar_mat_mul as ext


def check_correctness():
    print("=" * 60)
    print("1) Correctness vs torch.matmul")
    print("=" * 60)
    shapes = [(128, 128, 128), (256, 256, 256), (512, 512, 512),
              (1024, 1024, 1024), (2048, 2048, 2048), (256, 384, 512)]
    all_ok = True
    for (M, N, K) in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        c = ext.mat_mul(a, b)
        ref = a @ b
        err = (c - ref).abs().max().item()
        ok = err < 1e-2
        all_ok &= ok
        print(f"  {M:>5}x{N:<5}x{K:<5}  max_abs_err={err:.2e}  {'PASS' if ok else 'FAIL'}")
    print(f"  --> {'ALL PASS' if all_ok else 'SOME FAILED'}\n")
    return all_ok


def check_guards():
    print("=" * 60)
    print("2) Input-validation guards (each should raise)")
    print("=" * 60)
    a = torch.randn(128, 128, device="cuda")

    def expect_fail(desc, fn):
        try:
            fn()
            print(f"  {desc:<22}: NO ERROR  <-- guard missing!")
            return False
        except Exception as e:
            print(f"  {desc:<22}: raised OK  ({str(e).splitlines()[0][:50]})")
            return True

    ok = True
    ok &= expect_fail("non-aligned M=100", lambda: ext.mat_mul(
        torch.randn(100, 128, device="cuda"), torch.randn(128, 128, device="cuda")))
    ok &= expect_fail("cpu tensor", lambda: ext.mat_mul(
        torch.randn(128, 128), torch.randn(128, 128)))
    ok &= expect_fail("float64 dtype", lambda: ext.mat_mul(a.double(), a.double()))
    ok &= expect_fail("mismatched inner", lambda: ext.mat_mul(
        torch.randn(128, 256, device="cuda"), torch.randn(128, 128, device="cuda")))
    ok &= expect_fail("non-contiguous", lambda: ext.mat_mul(
        torch.randn(128, 256, device="cuda").t(), torch.randn(256, 128, device="cuda")))
    print(f"  --> {'ALL GUARDS OK' if ok else 'SOME GUARDS MISSING'}\n")
    return ok


def benchmark(size=2048, iters=50):
    print("=" * 60)
    print(f"3) Speed vs torch.matmul  (M=N=K={size}, {iters} iters)")
    print("=" * 60)
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    flop = 2.0 * size * size * size

    def timed(fn):
        for _ in range(3):  # warmup
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        return ms, flop / (ms * 1e6)  # ms, GFLOPS

    mine_ms, mine_gf = timed(lambda: ext.mat_mul(a, b))
    torch_ms, torch_gf = timed(lambda: a @ b)
    print(f"  nihar_mat_mul : {mine_ms:8.3f} ms   {mine_gf:9.1f} GFLOPS")
    print(f"  torch.matmul  : {torch_ms:8.3f} ms   {torch_gf:9.1f} GFLOPS")
    print(f"  --> torch is {mine_ms / torch_ms:.1f}x faster "
          f"(expected; it uses cuBLAS + tensor cores)\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available in this environment.")
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"device {torch.cuda.get_device_name(0)}\n")

    ok = check_correctness()
    ok &= check_guards()
    benchmark()

    print("=" * 60)
    print("RESULT:", "everything works" if ok else "SOMETHING FAILED")
    print("=" * 60)
