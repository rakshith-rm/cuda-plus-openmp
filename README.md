# cuda-plus-openmp

To run openMP + CUDA: nvcc -O3 -Xcompiler "-fopenmp" -o vector_add vector-add.cu main.cpp && ./vector_add

To run OpenMP + CUDA stream: nvcc -Xcompiler -fopenmp -o vector-add-hybrid vector-add.cu main.cpp && ./vector-add-hybrid

Result:
Time taken (Single CUDA stream): 0.727564 seconds
Result[123456] = 370368
Time taken (OpenMP + 1 CUDA stream per thread): 0.0828081 seconds
Result[123456] = 370368
Results match: Yes
