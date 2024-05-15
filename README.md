# Optimized MLP for MNIST Dataset using Auto-Differentiation

## Introduction
The motivation behind this project can be broken down into three main parts: computational, software engineering, and personal motivation. 

Auto Differentiation is part of every modern machine learning framework, as it creates great benefits for computational efficiency in training.  When dealing with an N-dimensional matrix of variables, without auto-differentiation, we would need to compute the derivative N-times for the Backward pass. We can ignore the forward pass due to the fact that with our simple MLP we are not actually calculating the gradient in the forward pass.

Simply put, Auto Differentiation is the mathematical technique, based on multiplication of Jacobian Matrices, for calculating the derivatives of each of these functions with only one backward pass. Manually teaching each of the operations “how” to calculate their own respective derivatives allows for massive computational benefits.

Furthermore, Auto Differentiation allows for increased numerical stability through all computations.  The other options for increasing computational efficiency include: Finite Differentiation and Symbolic Differentiation, yet each method has its own drawbacks.

In terms of Software Engineering, AutoDiff has quite a simple implementation compared to the alternative, with speeds in the upper echelon of the maximum speed possible to obtain. Furthermore, once implemented, it is extremely scalable to larger neural networks for obvious reasons. 

On a personal note, we have often used AutoDiff in modern Packages like PyTorch and Tensorflow, so it is extremely interesting to see how these useful tools implemented AutoDiff themselves. 

## Design and Implementation

###  Stack Implementation
We implemented a global stack so that each layer's forward pass is computed, followed by applying the ReLU activation function. The corresponding backward operation (op_relu_back) is pushed onto the “back_ops” stack for each ReLU operation. 
Now for the backward pass: For each layer, starting from the last layer, we pop and execute the backward operations from the “back_ops” stack. This is technically automatically creating the computational graph so that we store the derivatives according to the order in which the data flows through the graph. This ensures that we apply the gradients correctly. 

The backward operations for each ReLU activation are executed first (as they were the last operations to be pushed onto the stack during the forward pass), followed by the backward pass through each layer.


### Operations
Each operation within our framework, including Addition, Multiplication and Relu, has been changed in order to support AutoDiff. This involves manually teaching each of the operations how to compute its individual derivatives, which is stored in the global stack.  We define two primary methods within each of the operations.  
Forward Method: we calculate the results of the operation, and pushes the corresponding backwards operation to the “back_ops” tensor, keeping track of the computational graph
Backwards Method: We compute the gradients with respect to the chain rule and inputs, and then we update the tensor’s gradient attribute

For example, for the case of addition, the forward pass simply computes “a + b = c”, where ‘c’ is stored with the corresponding operation in the stack. Then for backprop, the gradients follow as ∂a ← ∂c, and ∂b ← ∂c. For more complicated multiplication, the gradients will propagate using the product rule. 

### Training Loop
Our training loop iterates through the dataset, where we perform the forward and backward pass for each batch. The difference between the old implementation is that, during the forward pass, the results are computed and their gradients are stacked. Once the output is computed for the nodes, we initiate the backward propagation, which simply sequentially pops from the “back_ops” stack and performs the operation stored. 

The training loop also includes the parameter update step, where we use a simple gradient descent approach: W ← W - lr * W.grad. To prevent the unnecessary expansion of the computation graph due to in-place tensor updates, we use the detach() function.

### Detach Function
This function is crucial for our implementation, as it ensures that our updating of the weights does not inadvertently expand the computational graph.  When we ‘detach’ a tensor, we basically remove its connection to the computational graph, which allows us to update its weights without changing the graph structure. This operation occurs after we compute the gradient, but before we update the weights in each of the tensors, ensuring we only focus on necessary calculations.  

