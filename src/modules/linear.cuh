#pragma once
#include "modules/param.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_mm.cuh"
#include <stack>
#include <functional>
#include <iostream>

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
        std::cout << "Initialized weights: " << w.t.str() << std::endl;
        std::cout << "Initialized biases: " << b.t.str() << std::endl;
    }

    // This function calculates the output of a linear layer and stores the result in tensor "y"
    void forward(const Tensor<float> &x, Tensor<float> &y) {
        std::cout << "Forward: x: " << x.h << "x" << x.w << ", w.t: " << w.t.h << "x" << w.t.w << ", y: " << y.h << "x" << y.w << std::endl;
        op_mm(x, w.t, y);
        std::cout << "Before op_add: y: " << y.h << "x" << y.w << ", b.t: " << b.t.h << "x" << b.t.w << std::endl;
        op_add(y, b.t, y);

        // Autodiff: Push the backward operations to the stack
        back_ops.push([this, &x, &y]() {
            // Compute gradients for weights and input
            Tensor<T> dx(x.h, x.w, x.on_device);
            Tensor<T> dw(w.t.h, w.t.w, w.t.on_device);

            // Gradients for w: dw = x^T * dy
            Tensor<T> x_t = x.transpose();
            std::cout << "Backward (mm): x_t: " << x_t.h << "x" << x_t.w << ", dy: " << y.h << "x" << y.w << ", dw: " << dw.h << "x" << dw.w << std::endl;
            op_mm(x_t, y, dw);

            // Store gradients
            std::cout << "Backward: x: " << x.h << "x" << x.w << ", dw: " << dw.h << "x" << dw.w << std::endl;
            std::cout << "dw: " << dw.str() << std::endl;
            op_add(w.dt, dw, w.dt);

            // Accumulate gradients for b
            Tensor<T> db(1, y.w, y.on_device);
            op_sum(y, db);
            std::cout << "db: " << db.str() << std::endl;
            op_add(b.dt, db, b.dt);
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
        Tensor<T> w_t = w.t.transpose();
        std::cout << "Backward (mm): dy: " << dy.h << "x" << dy.w << ", w_t: " << w_t.h << "x" << w_t.w << ", dx: " << dx.h << "x" << dx.w << std::endl;
        op_mm(dy, w_t, dx);
    }
};
