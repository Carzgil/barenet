#pragma once
#include "modules/linear.cuh"
#include "ops/op_elemwise.cuh"
template <typename T>
class MLP
{
private:
    std::vector<LinearLayer<T>> layers;
    std::vector<int> layer_dims;
    std::vector<Tensor<T>> activ;
    std::vector<Tensor<T>> d_activ;

    int batch_size;
    int in_dim;
    bool on_gpu;

public:
    MLP(int batch_size_, int in_dim_, std::vector<int> layer_dims_, bool gpu)
        : batch_size(batch_size_), in_dim(in_dim_), layer_dims(layer_dims_)
    {
        for (int i = 0; i < layer_dims.size(); i++)
        {
            if (i == 0)
            {
                layers.emplace_back(in_dim, layer_dims[i], gpu);
            }
            else
            {
                layers.emplace_back(layer_dims[i - 1], layer_dims[i], gpu);
            }
        }
        // make all the activation tensors
        activ.reserve(layer_dims.size() - 1);
        d_activ.reserve(layer_dims.size() - 1);
        for (int i = 0; i < layer_dims.size() - 1; i++)
        {
            activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
            // technically, i do not need to save d_activ for backprop, but since iterative
            // training does repeated backprops, reserving space for these tensor once is a good idea
            d_activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
        }
    }

    std::vector<Parameter<T> *> parameters()
    {
        std::vector<Parameter<T> *> params;
        for (int i = 0; i < layer_dims.size(); i++)
        {
            auto y = layers[i].parameters();
            params.insert(params.end(), y.begin(), y.end());
        }
        return params;
    }

    void init() {
        for (int i = 0; i < layers.size(); i++) {
            layers[i].init_uniform();
        }
    }

    //This function peforms the forward operation of a MLP model
    //Specifically, it should call the forward oepration of each linear layer 
    //Except for the last layer, it should invoke Relu activation after each layer.
    void forward(const Tensor<T> &in, Tensor<T> &out)
    {
        for (int i = 0; i < layers.size(); i++) {   
            if (i == 0) {  
                layers[i].forward(in, activ[i]);
                op_relu(activ[i], activ[i]);
               
            }
            else if (i == layers.size() - 1) {
                layers[i].forward(activ[i - 1], out);
            }
            else {  
                layers[i].forward(activ[i - 1], activ[i]);
                op_relu(activ[i], activ[i]);
            }
        }
    }

    //This function perofmrs the backward operation of a MLP model.
    //Tensor "in" is the gradients for the outputs of the last linear layer (aka d_logits from op_cross_entropy_loss)
    //Invoke the backward function of each linear layer and Relu from the last one to the first one.
    void backward(const Tensor<T> &in, const Tensor<T> &d_out, Tensor<T> &d_in) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            if (i == layers.size() - 1) {   
                layers[i].backward(activ[i-1], d_out, d_activ[i-1]);
            } else {
                op_relu_back(activ[i], d_activ[i], d_activ[i]);
                if(i == 0) {
                    layers[i].backward(in, d_activ[i], d_in);
                } else {
                    layers[i].backward(activ[i - 1], d_activ[i], d_activ[i - 1]);
                }
            }
        }
    }
    
};
