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
    LinearLayer(int in_dim_, int out_dim_, bool gpu)
        : in_dim(in_dim_), out_dim(out_dim_), w(Parameter<T>{in_dim, out_dim, gpu}),
          b(Parameter<T>{1, out_dim, gpu}) {}

    LinearLayer() {}

    LinearLayer(LinearLayer&& other)
        : in_dim(other.in_dim), out_dim(other.out_dim), w(std::move(other.w)), b(std::move(other.b)) {}

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

    // This function calculates the output of a linear layer and stores the result in tensor "y"
    void forward(const Tensor<float> &x, Tensor<float> &y) {
        op_mm(x, w.t, y);
        op_add(y, b.t, y);

        // Autodiff: Push the backward operations to the stack
        back_ops.push([this, &x, &y]() {
            // Compute gradients for weights and input
            Tensor<T> dx(x.h, x.w, x.on_device);
            Tensor<T> dw(w.t.h, w.t.w, w.t.on_device);

            // Gradients for w: dw = x^T * dy
            Tensor<T> x_t(x.w, x.h, x.on_device);
            op_transpose(x, x_t);
            op_mm(x_t, y, dw);

            // Gradients for x: dx = dy * w^T
            Tensor<T> w_t(w.t.w, w.t.h, w.t.on_device);
            op_transpose(w.t, w_t);
            op_mm(y, w_t, dx);

            // Store gradients
            op_add(w.dt, dw, w.dt);
            op_add(b.dt, y, b.dt);
        });
    }

    // This function performs the backward operation of a linear layer
    // It computes the weight gradients (dw, db) and saves them in w.dt and b.dt respectively
    // It also computes the gradients of "x" and saves it in dx.
    void backward(const Tensor<float> &x, const Tensor<float> &dy, Tensor<float> &dx) {
        // Autodiff: Pop and execute the backward operation
        back_ops.top()();
        back_ops.pop();

        // Compute gradients for input
        Tensor<T> w_t(w.t.w, w.t.h, w.t.on_device);
        op_transpose(w.t, w_t);
        op_mm(dy, w_t, dx);
    }
};
