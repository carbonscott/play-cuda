/* 
 * Compile: nvcc -o stream_wait_event_example stream_wait_event_example.cu && nsys profile -o profile_result  --force-overwrite true ./stream_wait_event_example
 * 
 * This example code demonstrates cross-stream timing where:
 * 1. Events are recorded on stream 0
 * 2. Graph executes on a different stream (let's call it graph stream)
 * 3. cudaStreamWaitEvent ensures stream 0 waits for graph completion
 * 4. Final event records the true end time
 * 
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Use config flags */
#define ENABLE_CPU_TIMING 0 /* Set to 0 to disable timing measurements */
const int MATRIX_SIZE = 128; /* N for NxN matrices */
const int NUM_KERNELS = 50; /* How many times to launch the kernel in the graph */
const int NUM_LAUNCHES = 5; /* How many times to launch the graph */

__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    printf("Main Starts...\n");
    printf("Matrix Size: %dx%d, Kernels per Graph: %d, Graph Launches: %d\n", MATRIX_SIZE, MATRIX_SIZE, NUM_KERNELS, NUM_LAUNCHES);
    printf("=============================================================\n");

    /* Allocate device memory for matrices */
    size_t bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    /* Initialize matrices with simple values for testing */
    cudaMemset(d_A, 1, bytes);
    cudaMemset(d_B, 1, bytes);
    cudaMemset(d_C, 0, bytes);

    /* Create events for cross-stream synchronization and optional timing */
    cudaEvent_t kernelDoneEvent;
    cudaEventCreate(&kernelDoneEvent);

#if ENABLE_CPU_TIMING
    cudaEvent_t begEvent, endEvent;
    cudaEventCreate(&begEvent);
    cudaEventCreate(&endEvent);
#endif

    /* Create streams */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Graph setup */
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphCreate(&graph, 0);

    /* Setup kernel launch parameters */
    dim3 blockSize(16, 16);
    dim3 gridSize((MATRIX_SIZE + blockSize.x - 1) / blockSize.x, 
                  (MATRIX_SIZE + blockSize.y - 1) / blockSize.y);

    /* Create multiple kernel nodes in the graph */
    cudaGraphNode_t kernelNodes[NUM_KERNELS];
    cudaGraphNode_t* prevNode = NULL;

    for (int i = 0; i < NUM_KERNELS; i++) {
        cudaKernelNodeParams kernelParams = {0};
        kernelParams.func = (void*)matmul;
        kernelParams.gridDim = gridSize;
        kernelParams.blockDim = blockSize;
        kernelParams.sharedMemBytes = 0;

        void* args[] = {(void*)&d_A, (void*)&d_B, (void*)&d_C, (void*)&MATRIX_SIZE};
        kernelParams.kernelParams = args;

        /* Add kernel node with dependency on previous node (sequential execution) */
        const cudaGraphNode_t* dependencies = (i == 0) ? NULL : prevNode;
        size_t numDependencies = (i == 0) ? 0 : 1;

        cudaGraphAddKernelNode(&kernelNodes[i], graph, dependencies, numDependencies, &kernelParams);
        prevNode = &kernelNodes[i];
    }

    /* Instantiate graph */
    cudaGraphInstantiate(&graphExec, graph, 0);

    /* Per-launch execution to find overheads with optional timing */
#if ENABLE_CPU_TIMING
    for (int launch = 0; launch < NUM_LAUNCHES; launch++) {
        cudaEventRecord(begEvent, 0);               /* Record start on stream 0 (timing only) */

        cudaGraphLaunch(graphExec, stream);         /* Launch graph on different stream */
        cudaEventRecord(kernelDoneEvent, stream);   /* Record completion on graph stream */
        cudaStreamWaitEvent(0, kernelDoneEvent, 0); /* Make stream 0 wait for graph completion */

        cudaEventRecord(endEvent, 0);               /* Record end on stream 0 (timing only) */
        cudaEventSynchronize(endEvent);             /* Ask CPU to wait for the finish of endEvent */

        float launch_time;
        cudaEventElapsedTime(&launch_time, begEvent, endEvent);
        printf("Launch %d: %.3f ms\n", launch, launch_time);
    }
#else
    for (int launch = 0; launch < NUM_LAUNCHES; launch++) {
        cudaGraphLaunch(graphExec, stream);         /* Launch graph on different stream */
        cudaEventRecord(kernelDoneEvent, stream);   /* Record completion on graph stream */
        cudaStreamWaitEvent(0, kernelDoneEvent, 0); /* Make stream 0 wait for graph completion */
    }
#endif

    /* Cleanup */
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaEventDestroy(kernelDoneEvent);
#if ENABLE_CPU_TIMING
    cudaEventDestroy(begEvent);
    cudaEventDestroy(endEvent);
#endif
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
