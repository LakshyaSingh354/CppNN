#include <iostream>
#include "CNNModel.h"
#include "Tensor.h"

int main() {
    CNNModel cnn;

    // Create a random input tensor with the correct dimensions
    Tensor input(28, 28, 1);  // Example: 28x28 grayscale image
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            input.at(i, j, 0) = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Forward pass
    Tensor output = cnn.forward(input);
    std::cout << "Output from CNN Model (forward pass):" << std::endl;
    for (int i = 0; i < output.getWidth(); i++) {
        std::cout << output.at(i, 0, 0) << " ";
    }
    std::cout << std::endl;

    // Create a random gradient output tensor for backward pass
    Tensor grad_output(output.getWidth(), 1, 1);
    for (int i = 0; i < grad_output.getWidth(); i++) {
        grad_output.at(i, 0, 0) = 1.0;  // Example gradient value
    }

    // Backward pass
    cnn.backward(grad_output);

    // Update weights with a learning rate
    cnn.updateWeights(0.01);

    return 0;
}