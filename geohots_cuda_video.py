import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

# dims = G*L (g*num_threads + l)
#            OpenCL        CUDA       HIP
# dims: total number of threads
# G cores   (get_group_id, blockIdx,  __ockl_get_group_id, threadgroup_position_in_grid)
# L threads (get_local_id, threadIdx, __ockl_get_local_id, thread_position_in_threadgroup)

# GPUs have warps. Warps are groups of threads, and all modern GPUs have them as 32 threads.
# GPUs are multicore processors with 32 threads

# On NVIDIA, cores are streaming multiprocessors.
#   AD102 (4090) has 144 SMs with 128 threads each
# On AMD, cores are compute units.
#   7900XTX has 96 CUs with with 64 threads each
# On Apple, ???
#   M3 Max has a 40 core GPU, 640 "Execution Units", 5120 "ALUs"
#   Measured. 640 EUs with 32 threads each (20480 threads total)

# SIMD - Single Instruction Multiple Data
#    vector registers
#    float<32> (1024 bits)
#    c = a + b (on vector registers, this is a single add instruction on 32 pieces of data)

# SIMT - Single Instruction Multiple Thread
#    similar to SIMD, but load/stores are different
#    you only declare "float", but behind the scenes it's float<32>
#    load stores are implicit scatter gather, whereas on SIMD it's explicit

kernel_code = r'''
extern "C" __global__
void add_kernel(float* c, int num_iterations) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a = (float)tid;

    // Heavy computational loop to simulate workload
    for (int i = 0; i < num_iterations; i++) {
        a *= 2.0f;
    }

    // Store results
    c[gid] = (float)tid;
    c[gid + 128] = a;
}
'''

def main():
    # Initialize CUDA kernel
    add_kernel = cp.RawKernel(kernel_code, 'add_kernel')

    # Get device info
    device_properties = cp.cuda.runtime.getDeviceProperties(0)
    print(f"Using GPU: {device_properties['name'].decode()}")
    print(f"Max threads per block: {device_properties['maxThreadsPerBlock']}")
    max_threads_per_block = device_properties['maxThreadsPerBlock']

    # Number of iterations for the loop
    num_iterations = 1000000

    # Warmup
    c_gpu = cp.zeros(32768, dtype=cp.float32)
    add_kernel((1,), (1,), (c_gpu, num_iterations))
    cp.cuda.stream.get_current_stream().synchronize()

    # Setup for plotting
    plt.figure(figsize=(10, 8))
    plt.title("CuPy Performance by Block Size", fontsize=16)
    plt.xlabel("Total Threads", fontsize=14)
    plt.ylabel("Execution Time (microseconds)", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Colors for different block sizes
    colors = ['red', 'magenta', 'cyan', 'green', 'blue', 'purple', 'orange']

    # Test different block sizes (threads per block)
    block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    max_global_size = 40960  # Similar limit to original code

    for idx, block_size in enumerate(block_sizes):
        if block_size > max_threads_per_block:
            print(f"Skipping block size {block_size} as it exceeds device limit")
            continue

        points = []
        color = colors[idx % len(colors)]

        # Test with increasing grid sizes
        for total_threads in range(64, min(max_global_size, 640*2*block_size), max(64, block_size)):
            # Calculate grid size (number of blocks)
            grid_size = (total_threads + block_size - 1) // block_size
            actual_threads = grid_size * block_size

            # Reset device array
            c_gpu = cp.zeros(32768, dtype=cp.float32)

            # Measure execution time
            start_time = time.time()

            # Launch kernel
            add_kernel((grid_size,), (block_size,), (c_gpu, num_iterations))
            cp.cuda.stream.get_current_stream().synchronize()

            elapsed = (time.time() - start_time) * 1_000_000  # Convert to microseconds
            print(f"Elapsed: {elapsed:.2f} microseconds for {actual_threads} threads with {block_size} thread block size")
            points.append((actual_threads, elapsed))

        # Plot this series
        if points:
            x_vals, y_vals = zip(*points)
            plt.plot(x_vals, y_vals, marker='o', linestyle='-', color=color, 
                     label=f'Block Size: {block_size}')

    # Display some results
    c_cpu = cp.asnumpy(c_gpu[:128])
    print("\nSample results:")
    for i in range(0, min(128, len(c_cpu)), 16):
        print(" ".join(f"{int(c_cpu[i+j]):<3d}" for j in range(16)))

    # Save figure
    plt.legend()
    plt.savefig('cupy_performance_plot.png')
    plt.show()

    # Additional analysis - calculate efficiency by thread configuration
    print("\nEfficiency Analysis:")

    # Select a reference point (usually the fastest configuration)
    reference_block_size = 32  # Typically efficient on NVIDIA GPUs
    reference_total = 1024     # A reasonable workload size

    # Run reference benchmark
    c_gpu = cp.zeros(32768, dtype=cp.float32)
    grid_size = (reference_total + reference_block_size - 1) // reference_block_size

    start_time = time.time()
    add_kernel((grid_size,), (reference_block_size,), (c_gpu, num_iterations))
    cp.cuda.stream.get_current_stream().synchronize()
    reference_time = (time.time() - start_time) * 1_000_000

    print(f"Reference configuration: {reference_total} threads with {reference_block_size} thread block size")
    print(f"Reference time: {reference_time:.2f} microseconds")

    # Plot efficiency graph (threads per time) to show which config is most efficient
    plt.figure(figsize=(10, 8))
    plt.title("Thread Efficiency by Block Size", fontsize=16)
    plt.xlabel("Block Size", fontsize=14)
    plt.ylabel("Relative Efficiency", fontsize=14)
    plt.grid(True, alpha=0.3)

    efficiencies = []

    for block_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if block_size > max_threads_per_block:
            continue

        # Use a fixed total of 1024 threads for comparison
        total_threads = 1024
        grid_size = (total_threads + block_size - 1) // block_size

        c_gpu = cp.zeros(32768, dtype=cp.float32)

        start_time = time.time()
        add_kernel((grid_size,), (block_size,), (c_gpu, num_iterations))
        cp.cuda.stream.get_current_stream().synchronize()
        elapsed = (time.time() - start_time) * 1_000_000

        # Calculate relative efficiency (higher is better)
        efficiency = reference_time / elapsed
        efficiencies.append((block_size, efficiency))
        print(f"Block size {block_size}: efficiency = {efficiency:.2f}x")

    # Plot efficiency
    x_vals, y_vals = zip(*efficiencies)
    plt.bar(x_vals, y_vals, color='blue', alpha=0.7)
    plt.axhline(y=1.0, color='red', linestyle='--', label='Reference Efficiency')

    plt.savefig('cupy_efficiency_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
