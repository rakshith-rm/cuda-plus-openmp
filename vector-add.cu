#include <cuda_runtime.h>

__global__ void vectorAddKernel(float* A, float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

extern "C" {
void runCudaVectorAdd(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void runCudaVectorAddStreams(float* A, float* B, float* C, int N, int num_streams) {
    const int chunk_size = N / num_streams;
    cudaStream_t* streams = new cudaStream_t[num_streams];
    float **d_A = new float*[num_streams];
    float **d_B = new float*[num_streams];
    float **d_C = new float*[num_streams];
    
    // Create streams and allocate memory for each stream
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_A[i], chunk_size * sizeof(float));
        cudaMalloc(&d_B[i], chunk_size * sizeof(float));
        cudaMalloc(&d_C[i], chunk_size * sizeof(float));
    }
    
    // Process chunks in parallel using streams
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int current_chunk_size = (i == num_streams - 1) ? (N - offset) : chunk_size;
        
        // Copy data to device asynchronously
        cudaMemcpyAsync(d_A[i], A + offset, current_chunk_size * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B[i], B + offset, current_chunk_size * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Launch kernel asynchronously
        int threadsPerBlock = 256;
        int blocksPerGrid = (current_chunk_size + threadsPerBlock - 1) / threadsPerBlock;
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], current_chunk_size);
        
        // Copy result back to host asynchronously
        cudaMemcpyAsync(C + offset, d_C[i], current_chunk_size * sizeof(float), 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all streams
    cudaDeviceSynchronize();
    
    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    
    delete[] streams;
    delete[] d_A;
    delete[] d_B;
    delete[] d_C;
}
} 