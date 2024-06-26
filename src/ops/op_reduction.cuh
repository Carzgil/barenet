#pragma once

#include "utils/tensor.cuh"
#define REDUCTION_BLOCK_DIM 32

template <typename T>
class MaxAccumFunc
{
public:
    //This function compares input x with the current accumulated maximum value stored in accum
    //If x is bigger than accum, stores x in accum and stores x's index (ind_x) to ind_accum
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      if (x > accum) {
            accum = x;
            ind_accum = ind_x;
        }
    }
};

template <typename T>
class SumAccumFunc
{
public:
    //This function adds input x to the current accumulated sum value stored in accum
    //The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used. 
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
        accum += x;
    }
};

//This kernel function performs column-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_colwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < in.h)
	{
		T accum = Index(in, col, 0);
		int ind_accum = 0;
		for (int row = 1; row < in.w; ++row)
		{
			f(Index(in, col, row), row, accum, ind_accum);
		}
		if (!get_index)
		{
			Index(out, col, 0) = accum;
		}
		else
		{
			Index(out_index, col, 0) = ind_accum;
		}
	}
}

//This kernel function performs row-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_rowwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < in.w)
	{
		T accum = in.rawp[row * in.stride_w];
		int ind_accum = 0;

		for (int col = 1; col < in.h; ++col)
		{
			f(in.rawp[row * in.stride_w + col * in.stride_h], col, accum, ind_accum);
		}
		if (!get_index)
		{
			out.rawp[row * out.stride_w] = accum;
		}
		else
		{
			out_index.rawp[row * out_index.stride_w] = ind_accum;
		}
	}
}    

template <typename OpFunc, typename T>
void op_reduction_gpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<int> &out_index, bool get_index = false)
{
    int out_h = out.h;
	if (!get_index)
	{
		assert((out.h == 1 && in.w == out.w) || (out.w == 1 && in.h == out.h));
	}
	else
	{
		out_h = out_index.h;
		assert((out_index.h == 1 && in.w == out_index.w) || (out_index.w == 1 && in.h == out_index.h));
	}
	if (in.h > out_h)
	{
		op_reduction_kernel_rowwise<<<(in.w + REDUCTION_BLOCK_DIM - 1) / REDUCTION_BLOCK_DIM, REDUCTION_BLOCK_DIM >>>(f, in, out, out_index, get_index);
	}
	else
	{
		op_reduction_kernel_colwise<<<(in.h + REDUCTION_BLOCK_DIM - 1) / REDUCTION_BLOCK_DIM, REDUCTION_BLOCK_DIM>>>(f, in, out, out_index, get_index);
	}
}


template <typename T>
void op_sum(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    SumAccumFunc<T> f;
    if (in.on_device && out.on_device) {
        op_reduction_gpu(f, in, out, out_index, false);
    } else
        assert(0);
}

template <typename T>
void op_argmax(const Tensor<T> &in, Tensor<int> &out_index)
{
    Tensor<T> out;
    MaxAccumFunc<T> f;
    if (in.on_device && out_index.on_device) {
        op_reduction_gpu(f, in, out, out_index, true);
    } else
        assert(0);
}
