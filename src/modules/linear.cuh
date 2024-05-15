#pragma once
#include "modules/param.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"
#include <stack>
#include <functional>

template<typename T>
class LinearLayer {
private:
    int in_dim;
    int out_dim;

    Parameter<T> w;
    Parameter<T> b;

    // Stack to keep track of operations for auto differentiation
    std::stack<std::function<void()>> back_ops;

public:
    LinearLayer(int in_dim_, int out_dim_, bool gpu): in_dim(in_dim_), out_dim(out_dim_) {
        w = Parameter<T>{in_dim, out_dim, gpu};
        b = Parameter<T>{1, out_dim, gpu};
    }

    LinearLayer() {}

    LinearLayer(LinearLayer&& other) : in_dim(other.in_dim), out_dim(other.out_dim), w(other.w), b(other.b) {}

    std::vector<Parameter<T>*> parameters() {
        std::vector<Parameter<T> *> v;
        v.push_back(&w);
        v.push_back(&b);
        return v;
    }

    void init_uniform() {
        // Do Kaiming uniform
        float max = 1.0f / std::sqrt(in_dim);
        op_uniform_init(w.t, -max, max);
        op_uniform_init(b.t, -max, max);
    }

    // This function calculates the output of a linear layer 
    // and stores the result in tensor "y"
    void forward(const Tensor<float> &x, Tensor<float> &y) {
        op_mm(x, w.t, y);
        op_add(y, b.t, y);

        // Record operations for the backward pass
        back_ops.push([this, x, y]() {
            op_mm(y, w.t.transpose(), x);  // Gradient for input x
            op_mm(x.transpose(), y, w.dt); // Gradient for weights w
            op_sum(y, b.dt);               // Gradient for bias b
        });
    }

    // This function performs the backward operation of a linear layer
    // Suppose y = Linear(x). Then function argument "dy" is the gradients of "y", 
    // and function argument "x" is the saved x.
    // This function computes the weight gradients (dw, db) and saves them in w.dt and b.dt respectively
    // It also computes the gradients of "x" and saves it in dx.
    void backward(const Tensor<float> &x, const Tensor<float> &dy, Tensor<float> &dx) {
        // Execute the recorded operations in reverse order
        while (!back_ops.empty()) {
            back_ops.top()();
            back_ops.pop();
        }
    }
};
