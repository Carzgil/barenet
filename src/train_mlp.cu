#include <getopt.h>

#include "modules/mlp.cuh"
#include "modules/linear.cuh"
#include "modules/sgd.cuh"

#include "utils/dataset_mnist.hh"
#include "ops/op_elemwise.cuh"
#include "ops/op_cross_entropy.cuh"
#include <iostream>

unsigned long long randgen_seed = 1;
static bool on_gpu = true;

int usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "\t-s <seed>      : Seed for RNG\n"
              << "\t-g <0|1>       : GPU usage flag (0 for CPU, 1 for GPU)\n"
              << "\t-h <dimensions>: Hidden dimensions\n"
              << "\t-l <layers>    : Number of layers\n"
              << "\t-b <batch size>: Batch size\n"
              << "\t-e <epochs>    : Number of epochs\n";
    return 1;  
}

int correct(const Tensor<float> &logits, const Tensor<char> &targets) {
    Tensor<int> predictions{targets.h, 1, on_gpu};
    op_argmax(logits, predictions);
    Tensor<int> correct_preds{1, 1, on_gpu};  // To accumulate the number of correct predictions
    op_equal(predictions, targets, correct_preds);
    int sum_correct = correct_preds.toHost().at(0, 0);  // Assuming a simple host transfer method
    return sum_correct;
}

void do_one_epoch(MLP<float>& mlp, SGD<float>& sgd, Tensor<float>& images, const Tensor<char>& targets, int batch_size, bool is_training, int epoch_num) {
    int num_batches = 0, total_correct = 0;
    float total_loss = 0.0;

    for (int b = 0; b < images.h / batch_size; ++b) {
        Tensor<float> batch_images = images.slice(b * batch_size, std::min((b + 1) * batch_size, images.h), 0, images.w);
        Tensor<char> batch_targets = targets.slice(b * batch_size, std::min((b + 1) * batch_size, targets.h), 0, targets.w);

        Tensor<float> logits{batch_size, 10, on_gpu};
        mlp.forward(batch_images, logits);
        Tensor<float> d_logits{batch_size, 10, on_gpu};
        float loss = op_cross_entropy_loss(logits, batch_targets, d_logits);
        total_loss += loss;
        total_correct += correct(logits, batch_targets);

        if (is_training) {
            Tensor<float> d_input_images{batch_size, images.w, on_gpu};
            mlp.backward(d_logits);
            sgd.step();
        }
        num_batches++;
    }

    std::cout << (is_training ? "TRAINING" : "TEST") << " epoch=" << epoch_num
              << " loss=" << total_loss / num_batches
              << " accuracy=" << static_cast<float>(total_correct) / (num_batches * batch_size)
              << " num_batches=" << num_batches << std::endl;
}

void train_and_test(int epochs, int batch_size, int hidden_dim, int n_layers) {
    MNIST mnist_train{"../data/MNIST/raw", MNIST::Mode::kTrain};
    MNIST mnist_test{"../data/MNIST/raw", MNIST::Mode::kTest};

    auto train_images = mnist_train.images.toDevice();
    auto train_targets = mnist_train.targets.toDevice();
    auto test_images = mnist_test.images.toDevice();
    auto test_targets = mnist_test.targets.toDevice();

    MLP<float> mlp{batch_size, mnist_train.images.w, {hidden_dim, 10}, on_gpu};
    mlp.init();
    SGD<float> sgd{mlp.parameters(), 0.01};

    for (int epoch = 0; epoch < epochs; ++epoch) {
        do_one_epoch(mlp, sgd, train_images, train_targets, batch_size, true, epoch);
    }
    do_one_epoch(mlp, sgd, test_images, test_targets, batch_size, false, 0);
}

int main(int argc, char *argv[]) {
    int opt, hidden_dim = 16, n_layers = 2, batch_size = 32, num_epochs = 10;
    while ((opt = getopt(argc, argv, "s:g:h:l:b:e:")) != -1) {
        switch (opt) {
            case 's': randgen_seed = atoll(optarg); break;
            case 'g': on_gpu = atoi(optarg) != 0; break;
            case 'h': hidden_dim = atoi(optarg); break;
            case 'l': n_layers = atoi(optarg); break;
            case 'b': batch_size = atoi(optarg); break;
            case 'e': num_epochs = atoi(optarg); break;
            default: /* '?' */ return usage();
        }
    }
    train_and_test(num_epochs, batch_size, hidden_dim, n_layers);
    return 0;
}
