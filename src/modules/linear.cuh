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

    // Capture the necessary tensors by reference and create temporary tensors inside the lambda
    back_ops.push([this, &x, &y]() {
        Tensor<float> w_t_transposed = w.t.transpose();
        Tensor<float> x_transposed = x.transpose();

        // Ensure the dimensions of the temporary tensors match the expected dimensions
        Tensor<float> dx(y.h, w.t.h, w.t.on_device);  // Create a tensor for dx
        Tensor<float> dw(w.t.h, w.t.w, w.t.on_device);  // Create a tensor for dw
        Tensor<float> db(1, y.w, w.t.on_device);        // Create a tensor for db

        // Perform matrix multiplications for gradients
        op_mm(y, w_t_transposed, dx);  // Gradient for input x
        op_mm(x_transposed, y, dw);    // Gradient for weights w
        op_sum(y, db);                 // Gradient for bias b

        // Transfer tensors to host for safe memory access
        Tensor<float> dx_host = dx.toHost();
        Tensor<float> dw_host = dw.toHost();
        Tensor<float> db_host = db.toHost();
        Tensor<float> w_dt_host = w.dt.toHost();
        Tensor<float> b_dt_host = b.dt.toHost();

        // Update the gradients directly using the Index macro
        for (int i = 0; i < dw_host.h; ++i) {
            for (int j = 0; j < dw_host.w; ++j) {
                Index(w_dt_host, i, j) += Index(dw_host, i, j);
            }
        }
        for (int i = 0; i < db_host.h; ++i) {
            for (int j = 0; j < db_host.w; ++j) {
                Index(b_dt_host, i, j) += Index(db_host, i, j);
            }
        }

        // Transfer updated tensors back to device
        w_dt_host.toDevice(w.dt);
        b_dt_host.toDevice(b.dt);

        // Debug updated tensor values
        // std::cout << "Updated w.dt: " << w.dt.str() << std::endl;
        // std::cout << "Updated b.dt: " << b.dt.str() << std::endl;
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
