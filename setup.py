# setup.py
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Define C++ extensions
ext_modules = [
    Pybind11Extension(
        "info_efficiency_cpp",
        ["src/bindings/python_bindings.cpp"],
        include_dirs=[
            "src",
            pybind11.get_include(),
        ],
        libraries=["info_efficiency_cuda", "info_efficiency_simd", "info_efficiency_core"],
        library_dirs=["build/lib"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="info-efficiency",
    version="1.0.0",
    author="Market Microstructure Team",
    description="High-performance information efficiency analysis using SIMD and multi-GPU CUDA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "redis>=4.6.0",
        "psycopg2-binary>=2.9.0",
        "fastapi>=0.101.0",
        "uvicorn>=0.23.0",
        "aiohttp>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "flake8>=6.0.0",
        ],
        "cuda": [
            "cupy-cuda11x>=12.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
    ],
)
