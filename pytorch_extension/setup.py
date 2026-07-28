from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name = "nihar_mat_mul", # Name of the python package
    ext_modules=[
        CppExtension(
            "nihar_mat_mul._cpp", # Module name as import in python
            ["custom_ops.cpp"]
        ),
    ],

    cmdclass={
        "build_ext" : BuildExtension
    }
)


