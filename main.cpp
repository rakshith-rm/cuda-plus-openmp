#include <iostream>
#include <omp.h>
#include <cstdlib>

extern "C" void runCudaVectorAdd(float*, float*, float*, int);

int main() {
    const int N = 1 << 24;
    const int num_chunks = 4;
    const int chunk_size = N / num_chunks;

    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        runCudaVectorAdd(A + offset, B + offset, C + offset, chunk_size);
    }

    double end = omp_get_wtime();
    std::cout << "Time taken (OpenMP + CUDA): " << (end - start) << " seconds\n";
    std::cout << "Result[123456] = " << C[123456] << "\n";

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
} 