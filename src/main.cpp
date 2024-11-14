#include <iostream>
#include "Tensor.h"
#include "ConvLayer.h"
#include "MaxPool.h"

int main() {
    Tensor input(4, 4, 1);

    // Initialize the input with some values
    int counter = 1;
    for (int i = 0; i < input.getWidth(); i++) {
        for (int j = 0; j < input.getHeight(); j++) {
            input.at(i, j, 0) = counter++;
        }
    }

    MaxPool pool(2, 2);
    Tensor pooled_output = pool.forward(input);

    // Display the pooled output
    std::cout << "Pooled Output:" << std::endl;
    for (int i = 0; i < pooled_output.getWidth(); i++) {
        for (int j = 0; j < pooled_output.getHeight(); j++) {
            std::cout << pooled_output.at(i, j, 0) << " ";
        }
        std::cout << std::endl;
    }

    // Create a grad_output tensor for testing backward pass
    Tensor grad_output(pooled_output.getWidth(), pooled_output.getHeight(), pooled_output.getDepth());
    for (int i = 0; i < grad_output.getWidth(); i++) {
        for (int j = 0; j < grad_output.getHeight(); j++) {
            grad_output.at(i, j, 0) = 1; // Simple gradient of 1 for testing
        }
    }

    // Perform the backward pass
    Tensor grad_input = pool.backward(grad_output);

    std::cout << "Grad Input:" << std::endl;
    for (int i = 0; i < grad_input.getWidth(); i++) {
        for (int j = 0; j < grad_input.getHeight(); j++) {
            std::cout << grad_input.at(i, j, 0) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}