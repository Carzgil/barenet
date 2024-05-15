#pragma once
#include "utils/tensor.cuh"
#include "op_reduction.cuh"

template <typename T>
__global__ void accum_kernel(Tensor<T> logits, Tensor<T> out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < out.h and idy < logits.w) {
        float accum = 0.0;
        for(int i = 0; i < logits.w; i++) {   
            accum += exp(Index(logits, idx, i));
        }
        Index(out, idx, 0) = accum;
    }
    
    __syncthreads();

}

template <typename T>
__global__ void normalization_kernel(Tensor<T> logits, Tensor<int> xmax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if ( idx < logits.h && idy < logits.w) {
        int max_elem = Index(xmax, idx, 0);
        Index(logits, idx, idy) = Index(logits, idx, idy) - Index(logits, idx, max_elem);
    }
    __syncthreads();
}

template <typename T>
__global__ void softmax_kernel(Tensor<T> logits, Tensor<T> exp_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ( idx < logits.h && idy < logits.w) {
        Index(logits, idx, idy) = exp(Index(logits, idx, idy)) / Index(exp_sum, idx, 0);
    }

    __syncthreads();
}

template <typename T>
__global__ void cross_entropy_loss_kernel(Tensor<T> logits, Tensor<char> targets, Tensor<T> d_logits,Tensor<T> loss) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < logits.h) {
        for(int i = 0; i < logits.w; i++) {   
            int target = static_cast<int>(Index(targets, idx, 0));
            if(target == i) {
                Index(loss, idx, 0) = -logf(Index(logits, idx, i));
                Index(d_logits, idx, i) = (Index(logits, idx, i) - 1) / logits.h;
            } else {
                Index(d_logits, idx, i) = Index(logits, idx, i) / logits.h;
            }
        }
    }
    __syncthreads();
}

template <typename T>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<char> &targets, Tensor<T> &d_logits) {
    assert(logits.h == targets.h && logits.h == d_logits.h);
    assert(logits.w == d_logits.w);
    assert(targets.w == 1);

    assert(logits.on_device && targets.on_device && d_logits.on_device); 
    
    dim3 blockSize(32, 32);
    dim3 gridSize((logits.h + 32 - 1) / 32, (logits.w + 32 - 1) / 32);
    
    Tensor<int> x_max {logits.h, 1, true};
    Tensor<T> loss_compare{logits.h, 1, true};
    Tensor<T> accum{logits.h, 1, true};
    Tensor<T> loss{1, 1, true};

    op_argmax(logits, x_max);

    normalization_kernel<<<gridSize,blockSize>>>(logits, x_max);
    accum_kernel<<<gridSize,blockSize>>>(logits, accum);
    softmax_kernel<<<gridSize,blockSize>>>(logits, accum);
    cross_entropy_loss_kernel<<<gridSize,blockSize>>>(logits, targets, d_logits, loss_compare);
    
    op_sum(loss_compare, loss);
    Tensor<T> loss_h = loss.toHost();
    float tot_loss = Index(loss_h, 0, 0);

    // Check for NaNs in loss
    if (std::isnan(tot_loss) || std::isinf(tot_loss)) {
        std::cerr << "NaN or Inf in loss calculation" << std::endl;
    }
    
    return tot_loss / logits.h;
}
