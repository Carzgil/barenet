#pragma once

#include "utils/tensor.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <functional> 
#include <memory>  

#define ELEMWISE_BLOCK_DIM 32 // thread block has 32x32 threads

extern unsigned long long randgen_seed;

class RandGenGPU
{
public:
    RandGenGPU(unsigned long long s)
    {
        curandAssert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandAssert(curandSetPseudoRandomGeneratorSeed(gen, s));
    }
    curandGenerator_t gen;
};


//This functor calculates the SGD operation 
//given input t (one element of the parameter tensor) 
//and its gradient dt
template <typename T>
class SGDFunc
{
public:
    __host__ __device__ T operator()(T t, T dt)
    {
        T result = t - (dt * lr);
        return result; 
    }
    const float lr;
};

//This functor adds two input elements "a" and "b" together
template <typename T>
class AddFunc
{
public:
    __host__ __device__ T operator()(T a, T b)
    {
        return a + b;
    }
};

//This functor adds constant "b" to the input element
template <typename T>
class AddConstFunc
{
public:
    __host__ __device__ T operator()(T a)
    {
        return a + b;
    }
    const T b;
};

//This functor multiplies two input elements x and b together
template <typename T>
class MultiplyFunc
{
public:
    __host__ __device__ T operator()(T x, T a)
    {
        return x * a;

    }
};

//This functor multiplies constant "b" to the input element
template <typename T>
class MultiplyConstFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        return x * b;

    }
    const T b;
};

//This functor returns 1 if inputs "a" and "b" are equal
//and returns 0 otherwise. 
template <typename AT, typename BT, typename OutT>
class EqualityFunc
{
public:
    __host__ __device__ OutT operator()(AT a, BT b)
    {
        return a == b ? 1 : 0;
    }

};

//This functor implements the ReLu operation 
//for a single element
template <typename T>
class ReluFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        return x > 0 ? x : 0;
    }
};

//This functor implements the backwards of the 
//ReLu operation for a single element.
template <typename T>
class ReluBackFunc
{
public:
    __host__ __device__ T operator()(T x, T dy)
    {
        return x > 0 ? dy : 0;
    }
};

template <typename T>
class ConstInitFunc
{
public:
    __host__ __device__ T operator()(T x)
    {
        return val;
    }
    const float val;
};

template <typename T>
class UniformInitFuncCPU
{
public:
    UniformInitFuncCPU(T min, T max) : dist(min, max) {}
    T operator()(T x)
    {
        static std::default_random_engine gen(randgen_seed);
        return dist(gen);
    }
    std::uniform_real_distribution<T> dist;
};

//This is the GPU kernel function for performing element wise operation 
//that takes a single argument "t" and stores the result in "out"
template <typename OpFunc, typename T>
__global__ void op_elemwise_unary_kernel(OpFunc f, Tensor<T> t, Tensor<T> out)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;


    if(IndexOutofBound(out, row, col)){
        return;
    }
    else{
        T t_val = Index(t, row, col);
        Index(out, row, col) = f(t_val);
    }
}

//This function launches the GPU kernel to perform element wise operation 
//that takes a single argument "t" and stores the result in "out"
template <typename OpFunc, typename T>
void op_elemwise_unary_gpu(OpFunc f, const Tensor<T> &t, Tensor<T> &out)
{
    dim3 blockSize(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM);
    dim3 gridSize((out.h + ELEMWISE_BLOCK_DIM - 1) / ELEMWISE_BLOCK_DIM, (out.w + ELEMWISE_BLOCK_DIM - 1) / ELEMWISE_BLOCK_DIM);
    op_elemwise_unary_kernel<<<gridSize, blockSize>>>(f, t, out);
    
}

//This is the GPU kernel function for performing element wise operation with 
//two input arguments "in1" and "in2" with potential broadcasting. 
// Input tensor "in2" is always the one to be 
// broadcasted when broadcasting is necessary.  Broadcasting is needed if 
// "in2" only have one dimension (instead of both dimensions) in common with "in1" 
// and its other dimension has size 1. In this case, to perform elemwise operation,
// we essentially broadcast the values of "in2" along the dimension with size 1 
// to match the dimension size of "in1". 
// Example1: a = [[1, 2, 3], [4, 5, 6]] and b = [[10],[20]], 
// then a+b = [[11, 12, 13], [24, 25, 26]]
// Example2: a = [[1, 2, 3], [4, 5, 6]] and b = [[10,20,30]]
// then a+b = [[11,22,33], [14, 25, 36]] 
template <typename OpFunc, typename AT, typename BT, typename OutT>
__global__ void op_elemwise_binary_w_bcast_kernel(OpFunc f, Tensor<AT> in1, Tensor<BT> in2, Tensor<OutT> out)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col  = blockIdx.y * blockDim.y + threadIdx.y;
    if ( in2.w == 1 && in1.h == in2.h){
        if(IndexOutofBound(out, row, col)){
            return;
        }
        else{
            AT in1_val = Index(in1, row, col);
            BT in2_val = Index(in2, row, 0);
            Index(out, row, col) = f(in1_val,in2_val);
        }
    }
    if (in2.h == 1 && in1.w == in2.w){
        if(IndexOutofBound(out, row, col)){
            return;
        }
        else{
            AT in1_val = Index(in1, row, col);
            BT in2_val = Index(in2, 0, col);
            Index(out, row, col) = f(in1_val,in2_val);
        }
    }
    if (in2.h == in1.h && in2.w == in1.w){
        if(IndexOutofBound(out, row, col)){
            return;
        }
        else{
            AT in1_val = Index(in1, row, col);
            BT in2_val = Index(in2, row, col);
            Index(out, row, col) = f(in1_val,in2_val);
        }
    }

}

//This function launches the GPU kernel that performs elementwise operation 
//(with potential broadcast) with two input tensor arguments "in1" and "in2",
// and stores the result in "out".  
template <typename OpFunc, typename AT, typename BT, typename OutT>
void op_elemwise_binary_w_bcast_gpu(OpFunc f, const Tensor<AT> &in1, const Tensor<BT> &in2, Tensor<OutT> &out)
{
    assert(out.h == in1.h && out.w == in1.w);
    dim3 blockSize(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM);
    dim3 gridSize((out.h + ELEMWISE_BLOCK_DIM - 1) / ELEMWISE_BLOCK_DIM, (out.w + ELEMWISE_BLOCK_DIM - 1) / ELEMWISE_BLOCK_DIM);

    op_elemwise_binary_w_bcast_kernel<<<gridSize,blockSize>>>(f,in1,in2,out);

}

/*----------------------- tensor operators-----------------------*/

//This operator implements ReLu and stores the result in "out".
//Suppose y = Relu(x) Then y = x if x >=0.  y= 0 if x < 0.
template <typename T>
void op_relu(const Tensor<T> &input, Tensor<T> &output) {
    assert(output.h == input.h && output.w == input.w);
    ReluFunc<T> f;

    if (input.on_device && output.on_device) {
        op_elemwise_unary_gpu(f, input, output);

        // auto backward_op = [&input, &output]() {
        //     if (input.grad) { 
        //         op_relu_back(input, *output.grad, *input.grad);
        //     }
        // };

        // output.op = std::make_shared<Op<T>>(backward_op);
    } else {
        assert(0); 
    }
}

//This operator is the "backward" function of ReLu. Let out = ReLu(in).
//Let "d_out" represents the gradient of "out". Calculate the gradient 
//of "in" using the chain rule and store the result in "d_in".
template <typename T>
void op_relu_back(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in)
{
    assert(d_in.h == in.h && d_in.w == in.w);
    assert(in.h == d_out.h && in.w == d_out.w);
    ReluBackFunc<T> f;

    // Ensure the gradient tensor for d_in is properly initialized
    // if (!d_in.grad) {
    //     d_in.grad = std::make_shared<Tensor<T>>(in.h, in.w, in.on_device);
    //     op_const_init(*d_in.grad, 0.0f);
    // }

    if (d_in.on_device && in.on_device && d_out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, in, d_out, *d_in.grad);
    } else {
        assert(0);
    }
}

//This operator performs the "SGD" operation, aka calculating out = t - lr*dt;
//and stores the result in "out" tensor. lr is the learning rate. dt tensor should
//contain the gradient of parameter tensor "t".
template <typename T>
void op_sgd(const Tensor<T> &t, const Tensor<T> &dt, Tensor<T> &out, float lr)
{
    assert(out.h == t.h && out.w == t.w);
    assert(t.h == dt.h && t.w == dt.w);

    SGDFunc<T> f{lr};
    if (t.on_device && dt.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, t, dt, out);
    } else {
        assert(0);
    }
}

//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T>
void op_add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w) || (a.h == b.h && b.w == 1) || (a.w == b.w && b.h == 1));
    AddFunc<T> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);

        // auto backward_op = [&]() {
        //     if (a.grad) {
        //         op_add(*a.grad, *out.grad, *a.grad);
        //     }
        //     if (b.grad) {
        //         op_add(*b.grad, *out.grad, *b.grad);
        //     }
        // };
        // out.op = std::make_shared<Op<T>>(backward_op);
    } else {
        assert(0); 
    }
}


//This operator performs element-wise addition of "a" and constant b
//stores the result in tensor "out"
template <typename T>
void op_add(const Tensor<T> &a, T b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    AddConstFunc<T> f{b};
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);

        // auto backward_op = [&]() {
        //     if (a.grad) {
        //         op_add(*a.grad, *out.grad, *a.grad);
        //     }
        // };
        // out.op = std::make_shared<Op<T>>(backward_op);
    } else {
        assert(0); 
    }
}

//This operator performs element-wise multiplication of "a" and "b" and 
//stores the result in tensor "out"
template <typename T>
void op_multiply(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w) || (a.h == b.h && b.w == 1) || (a.w == b.w && b.h == 1));
    MultiplyFunc<T> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);

        // auto backward_op = [&]() {
        //     if (a.grad) {
        //         op_multiply(b, *out.grad, *a.grad);
        //     }
        //     if (b.grad) {
        //         op_multiply(a, *out.grad, *b.grad);
        //     }
        // };
        // out.op = std::make_shared<Op<T>>(backward_op);
    } else {
        assert(0); 
    }
}


//This operator performs element-wise multiplication of "a" and constant b
//stores the result in tensor "out"
template <typename T>
void op_multiply(const Tensor<T> &a, T b, Tensor<T> &out)
{
    assert(out.h == a.h && out.w == a.w);
    MultiplyConstFunc<T> f{b};
    if (a.on_device && out.on_device) {
        op_elemwise_unary_gpu(f, a, out);

        // auto backward_op = [&]() {
        //     if (a.grad) {
        //         op_multiply(*a.grad, b, *a.grad);
        //     }
        // };
        // out.op = std::make_shared<Op<T>>(backward_op);
    } else {
        assert(0); 
    }
}

//This operator checks if tensor "a" and "b" are the same
//and stores in the "out" tensor value 0 at places where "a" and "b" are not equal 
//and 1 at places where "a" and "b" are equal.
template <typename AT, typename BT, typename OutT>
void op_equal(const Tensor<AT> &a, const Tensor<BT> &b, Tensor<OutT> &out)
{
    assert(out.h == a.h && out.w == a.w);
    assert((a.h == b.h && a.w == b.w));
    EqualityFunc<AT, BT, OutT> f;
    if (a.on_device && b.on_device && out.on_device) {
        op_elemwise_binary_w_bcast_gpu(f, a, b, out);
    } else {
        assert(0);
    }
}

//This operator initializes tensor with constant values
template <typename T>
void op_const_init(Tensor<T> &t, float init_val)
{
    ConstInitFunc<float> f{init_val};
    if (t.on_device) {
        op_elemwise_unary_gpu(f, t, t);
    } else {
        for (int i = 0; i < t.h; i++) {
            for (int j = 0; j < t.w; j++) {
                Index(t, i, j) = init_val;
            }
        }
    }
}

//This operator initializes tensor with random values
//that are uniformly distributed between min and max
template <typename T>
void op_uniform_init(Tensor<T> &t, T min = 0, T max = 1)
{
    // XXX: Currently, this only works with un-sliced tensor
    assert(t.offset == 0 && t.stride_w == 1);
    if (t.on_device)
    {
        static RandGenGPU g(randgen_seed);
        //curandGenerateUniform generates elements in the range [0,1)
        curandAssert(curandGenerateUniform(g.gen, t.rawp, t.h * t.w));
        //scale the shift the elements to be in the range [min, max)
        op_add<T>(t, min/(max-min), t);
        op_multiply(t, max-min, t);
    } else {
        assert(0);
    }
}

// This operator checks if all elements of two tensors are the "same" (aka close enough) with each other
// For now, let's settle with only CPU implementation of allclose
template <typename T>
bool op_allclose(const Tensor<T> &at, Tensor<T> &bt)
{
    if (at.h != bt.h || at.w != bt.w)
    {
        return false;
    }
    Tensor<T> att;
    if (at.on_device) {
        att =  at.toHost();
    } else {
        att = at;
    }
    Tensor<T> btt;
    if (bt.on_device) {
        btt = bt.toHost();
    } else {
        btt = bt;
    }
    for (int i = 0; i < at.h; i++)
    {
        for (int j = 0; j < at.w; j++)
        {
            // Check if the numbers are close using both relative and absolute tolerances
            T a = Index(att, i, j);
            T b = Index(btt, i, j);
            if (std::abs(a - b) >
                std::max(ISCLOSE_RELTOL * std::max(std::abs(a), std::abs(b)), ISCLOSE_ABSTOL))
            {
                std::cout << "(" << i << "," << j << ") this=" << a << " other=" << b << " diff=" << (a - b) << std::endl;
                return false;
            }
        }
    }
    return true;
}
