#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define MM_BLOCK_DIM 32

//This operator compute C = A@B
template <typename T>
__global__ void matrixMulKernel(const Tensor<T> A, const Tensor<T> B, Tensor<T> C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < C.h && col < C.w) {
        T sum = 0;
        for (int k = 0; k < A.w; ++k) {
            sum += Index(A, row, k) * Index(B, k, col);
        }
        Index(C, row, col) = sum;
    }
}


template <typename T>
void op_mm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C)
{
    assert(A.h == C.h && B.w == C.w && A.w == B.h);
    assert(A.on_device && B.on_device && C.on_device);

    dim3 blockSize(MM_BLOCK_DIM, MM_BLOCK_DIM);
    dim3 gridSize((C.w + blockSize.x - 1) / blockSize.x, (C.h + blockSize.y - 1) / blockSize.y);

    matrixMulKernel<<<gridSize, blockSize>>>(A, B, C);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in matrix multiplication: " << cudaGetErrorString(error) << std::endl;
        assert(0);
    }

}

template <typename T>
void op_mm_backward(const Tensor<T>& dC, const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& dA, Tensor<T>& dB) {
    Tensor<T> B_transposed = B.transpose();
    op_mm(dC, B_transposed, dA);

    Tensor<T> A_transposed = A.transpose();
    op_mm(A_transposed, dC, dB);
}