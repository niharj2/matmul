from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name = "nihar_mat_mul", # Name of the python package
    ext_modules=[
        CUDAExtension(
            "nihar_mat_mul", # Module name as import in python
            [
                "custom_ops.cpp",
                "../cuda_SGEMM_implementation/2D_blocktiling/2D_blocktiling_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-arch=sm_90"],
            },
        ),
    ],

    cmdclass={
        "build_ext" : BuildExtension
    }
)


