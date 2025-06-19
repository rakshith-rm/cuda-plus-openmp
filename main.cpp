#include <iostream>
#include <cstdlib>
#include <omp.h>

extern "C" {
    void runCudaVectorAdd(float*, float*, float*, int);
    void runCudaVectorAddStreams(float*, float*, float*, int, int);
}

int main() {
    const int N = 1 << 24;
    const int num_chunks = 4;
    const int chunk_size = N / num_chunks;
    const int num_streams = 1; // 1 stream per OpenMP thread

    float* A = new float[N];
    float* B = new float[N];
    float* C1 = new float[N];
    float* C2 = new float[N];

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Test single stream version
    double start1 = omp_get_wtime();
    runCudaVectorAdd(A, B, C1, N);
    double end1 = omp_get_wtime();
    std::cout << "Time taken (Single CUDA stream): " << (end1 - start1) << " seconds\n";
    std::cout << "Result[123456] = " << C1[123456] << "\n";

    // Test OpenMP + CUDA streams version
    double start2 = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        int current_chunk_size = (i == num_chunks - 1) ? (N - offset) : chunk_size;
        runCudaVectorAddStreams(A + offset, B + offset, C2 + offset, current_chunk_size, num_streams);
    }
    
    double end2 = omp_get_wtime();
    std::cout << "Time taken (OpenMP + " << num_streams << " CUDA stream per thread): " << (end2 - start2) << " seconds\n";
    std::cout << "Result[123456] = " << C2[123456] << "\n";

    // Verify results match
    bool results_match = true;
    for (int i = 0; i < N; i++) {
        if (C1[i] != C2[i]) {
            results_match = false;
            break;
        }
    }
    std::cout << "Results match: " << (results_match ? "Yes" : "No") << "\n";

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    return 0;
} 