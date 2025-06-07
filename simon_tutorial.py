# =====================
# Setup and Environment
# =====================
import torch
import time
import math
import os
from torch.utils.cpp_extension import load_inline
import matplotlib.pyplot as plt
import numpy as np

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA must be available"

# Set environment variable for better error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set device and basic parameters
device = torch.device('cuda')
dtype = torch.float32

# Matrix dimensions for testing
M = N = K = 2048  # Start with 2048x2048 matrices

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Matrix dimensions: {M}x{N} @ {N}x{K} = {M}x{K}")

# Ensure build directory exists
def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)
# ===========================
# Benchmarking Infrastructure
# ===========================
def benchmark_kernel(kernel_func, A, B, num_runs=100, warmup=10):
    """Benchmark a matrix multiplication kernel"""
    # Warmup
    for _ in range(warmup):
        result = kernel_func(A, B)
    torch.cuda.synchronize()

    # Actual timing
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = kernel_func(A, B)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_runs

    # Calculate performance metrics
    total_ops = 2 * M * N * K  # FMA counts as 2 ops
    gflops = (total_ops / avg_time) / 1e9

    return avg_time, gflops, result

def verify_correctness(custom_result, reference_result, tolerance=1e-3):
    """Verify kernel correctness against reference"""
    diff = torch.abs(custom_result - reference_result)
    max_diff = torch.max(diff).item()
    relative_diff = max_diff / torch.max(torch.abs(reference_result)).item()

    is_correct = relative_diff < tolerance
    return is_correct, max_diff, relative_diff

# Benchmark cuBLAS as reference
def benchmark_cublas(A, B, num_runs=100):
    """Benchmark cuBLAS for comparison"""
    warmup = 10
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = torch.mm(A, B)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_runs
    total_ops = 2 * M * N * K
    gflops = (total_ops / avg_time) / 1e9

    return avg_time, gflops, result

# ==============================
# Kernel 1: Naive Implementation
# ==============================
# Create the naive kernel CUDA code
naive_cuda_code = """
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void sgemm_truly_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Truly naive implementation with poor memory coalescing
    // Maps threadIdx.x to rows and threadIdx.y to columns - intentionally bad for coalescing
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // Column index

    // When threads in a warp execute (with consecutive threadIdx.x values):
    // - They access different rows of A (stride K - non-coalesced)
    // - They access the same column but different rows of B (highly non-coalesced)
    // - They write to C with large strides (stride N - non-coalesced)

    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor sgemm_naive(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda());
    TORCH_CHECK(B.device().is_cuda());
    TORCH_CHECK(A.is_contiguous());
    TORCH_CHECK(B.is_contiguous());

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    // Allocate output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Launch configuration - maintain same block size for fair comparison
    dim3 block_size(16, 16);

    // Note: Grid dimensions are swapped compared to your previous implementation
    // because we've swapped the thread mapping (x=row, y=column)
    dim3 grid_size(cdiv(M, block_size.x), cdiv(N, block_size.y));

    // Launch kernel
    sgemm_truly_naive_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
"""

# C++ function declaration
naive_cpp_code = "torch::Tensor sgemm_naive(torch::Tensor A, torch::Tensor B);"

# Compile the kernel
ensure_directory_exists("var/cuda_matmul_naive")
naive_module = load_inline(
    name='naive_sgemm',
    cpp_sources=[naive_cpp_code],
    cuda_sources=[naive_cuda_code],
    functions=['sgemm_naive'],
    extra_cflags=['-O2'],
    verbose=True,
    build_directory="var/cuda_matmul_naive",
    extra_include_paths=[
        "/sdf/scratch/users/c/cwang31/miniconda/ana-4.0.59-torch/targets/x86_64-linux/include"
    ]
)

# Test the kernel
A = torch.randn(M, K, device=device, dtype=dtype)
B = torch.randn(K, N, device=device, dtype=dtype)

# Benchmark
naive_time, naive_gflops, C_naive = benchmark_kernel(naive_module.sgemm_naive, A, B)
cublas_time, cublas_gflops, C_reference = benchmark_cublas(A, B)

# Verify correctness
is_correct, max_diff, rel_diff = verify_correctness(C_naive, C_reference)

print(f"Kernel 1 (Naive):")
print(f"  Time: {naive_time*1000:.2f} ms")
print(f"  Performance: {naive_gflops:.1f} GFLOPS ({naive_gflops/cublas_gflops*100:.1f}% of cuBLAS)")
print(f"  Correctness: {'✓' if is_correct else '✗'} (max diff: {max_diff:.2e})")


# ==================================
# Kernel 2: Global Memory Coalescing
# ==================================
# Create the coalesced kernel CUDA code
coalesced_cuda_code = """
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void sgemm_coalesced_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Apply Simon Boehm's coalescing strategy:
    // Use 1D thread blocks and map threads to ensure warp-level coalescing

    const int BLOCK_SIZE = 32;  // Use 32 for perfect warp alignment
    int tid = threadIdx.x;      // Linear thread ID (0 to 1023)

    // Key insight: Map consecutive threads in a warp to consecutive columns
    // This ensures Matrix B and C accesses are coalesced, Matrix A is broadcast
    int row = blockIdx.y * BLOCK_SIZE + (tid / BLOCK_SIZE);  // Same row for consecutive 32 threads
    int col = blockIdx.x * BLOCK_SIZE + (tid % BLOCK_SIZE);  // Consecutive columns for consecutive threads

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Access pattern analysis for threads 0-31 in a warp:
    // - A[row * K + k]: All threads access same element → BROADCAST (optimal!)
    // - B[k * N + col]: Threads access B[k * N + 0], B[k * N + 1], ..., B[k * N + 31] → COALESCED!
    // - C[row * N + col]: Threads write C[row * N + 0], C[row * N + 1], ..., C[row * N + 31] → COALESCED!
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor sgemm_coalesced(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda());
    TORCH_CHECK(B.device().is_cuda());
    TORCH_CHECK(A.is_contiguous());
    TORCH_CHECK(B.is_contiguous());

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    torch::Tensor C = torch::zeros({M, N}, A.options());

    const int BLOCK_SIZE = 32;

    // Key change: Use 1D thread blocks for proper warp alignment
    // Each block processes a 32x32 tile of the output matrix
    dim3 block_size(BLOCK_SIZE * BLOCK_SIZE);  // 1024 threads in 1D layout
    dim3 grid_size(cdiv(N, BLOCK_SIZE), cdiv(M, BLOCK_SIZE));

    sgemm_coalesced_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}
"""

coalesced_cpp_code = "torch::Tensor sgemm_coalesced(torch::Tensor A, torch::Tensor B);"

ensure_directory_exists("var/cuda_matmul_coalesced")
coalesced_module = load_inline(
    name='coalesced_sgemm',
    cpp_sources=[coalesced_cpp_code],
    cuda_sources=[coalesced_cuda_code],
    functions=['sgemm_coalesced'],
    extra_cflags=['-O2'],
    verbose=True,
    build_directory="var/cuda_matmul_coalesced",
    extra_include_paths=[
        "/sdf/scratch/users/c/cwang31/miniconda/ana-4.0.59-torch/targets/x86_64-linux/include"
    ]
)

# Test and benchmark
coalesced_time, coalesced_gflops, C_coalesced = benchmark_kernel(coalesced_module.sgemm_coalesced, A, B)
is_correct, max_diff, rel_diff = verify_correctness(C_coalesced, C_reference)

print(f"\nKernel 2 (Coalesced):")
print(f"  Time: {coalesced_time*1000:.2f} ms")
print(f"  Performance: {coalesced_gflops:.1f} GFLOPS ({coalesced_gflops/cublas_gflops*100:.1f}% of cuBLAS)")
print(f"  Speedup over naive: {coalesced_gflops/naive_gflops:.1f}x")
print(f"  Correctness: {'✓' if is_correct else '✗'} (max diff: {max_diff:.2e})")
