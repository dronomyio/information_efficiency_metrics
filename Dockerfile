# Dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libcurl4-openssl-dev \
    libomp-dev \
    python3-dev \
    python3-pip \
    libzstd-dev \
    libarrow-dev \
    libparquet-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Intel oneAPI for advanced SIMD support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \
    echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2024.0 && \
    rm -rf /var/lib/apt/lists/*

# Set up Intel compiler environment
RUN echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    pyarrow==12.0.1 \
    pybind11==2.11.1 \
    pytest==7.4.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    requests==2.31.0 \
    aiohttp==3.8.5 \
    asyncio==3.4.3 \
    tqdm==4.65.0 \
    numba==0.57.1 \
    cupy-cuda11x==12.2.0

# Create working directory
WORKDIR /app

# Copy source files
COPY . /app/

# Build the project
RUN mkdir build && cd build && \
    cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" \
    .. && \
    ninja -j$(nproc)

# Install Python package
RUN pip3 install -e .

# Set up environment for multi-GPU
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV OMP_NUM_THREADS=32
ENV MKL_NUM_THREADS=32

# Expose port for API
EXPOSE 8080

# Default command
CMD ["python3", "-m", "info_efficiency.api"]
