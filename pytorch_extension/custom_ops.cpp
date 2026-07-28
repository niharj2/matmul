#include <torch/extension.h>

// This is the CUDA forward declaration
void launch_2d(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta);

// C++ interface function adhering to PyTorch's C++ API
torch::Tensor mat_mul(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");
    TORCH_CHECK(a.dim() == 2, "Input tensor a must be 2-D, got ", a.dim(), "-D");
    TORCH_CHECK(b.dim() == 2, "Input tensor b must be 2-D, got ", b.dim(), "-D");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Inner dimensions must match: a is ", a.size(0), "x", a.size(1),
                ", b is ", b.size(0), "x", b.size(1));

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    // launch_2d instantiates the kernel as <128,128,16,8>, which has no bounds
    // checks. Reject bad shapes here so Python gets an exception instead of the
    // exit() inside launch_2d killing the interpreter.
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 16 == 0,
                "the 2D blocktiling kernel requires M%128==0, N%128==0, K%16==0; got M=",
                M, " N=", N, " K=", K);

    float alpha = 1.0f;
    float beta  = 0.0f;

    // zeros, not empty: the kernel reads C for the beta*C term, and 0*NaN is NaN.
    // a.options() inherits dtype AND device, so multi-GPU inputs stay put.
    torch::Tensor c = torch::zeros({M, N}, a.options());

    launch_2d(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
              a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
              alpha, beta);

    return c;
}

// Binding function: expose the C++ mat_mul as 'mat_mul' in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mat_mul", &mat_mul, "SGEMM using a 2D block-tiled CUDA kernel");
}
