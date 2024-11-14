#include "DenseLayer.h"
#include <cmath>
#include <random>

using namespace std;

DenseLayer::DenseLayer(int inputSize, int outputSize)
    : inputSize(inputSize), outputSize(outputSize), weights(outputSize, inputSize, 1), bias(outputSize, 1, 1) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weights.at(j, i, 0) = dis(gen);
        }
        bias.at(i, 0, 0) = dis(gen);
    }
}

Tensor DenseLayer::forward(const Tensor &input) {
    inputCache = input;
    Tensor output(outputSize, 1, 1);

    for (int i = 0; i < outputSize; i++) {
        float sum = bias.at(i, 0, 0);
        for (int j = 0; j < inputSize; j++) {
            sum += input.at(j, 0, 0) * weights.at(i, j, 0);
        }
        output.at(i, 0, 0) = sum;
    }

    return output;
}

Tensor DenseLayer::backward(const Tensor &grad) {
    gradWeights = Tensor(outputSize, inputSize, 1);
    gradBias = Tensor(outputSize, 1, 1);
    Tensor gradInput(inputSize, 1, 1);

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            gradWeights.at(i, j, 0) += grad.at(i, 0, 0) * inputCache.at(j, 0, 0);
        }
        gradBias.at(i, 0, 0) += grad.at(i, 0, 0);
    }

    for (int j = 0; j < inputSize; j++) {
        float sum = 0.0;
        for (int i = 0; i < outputSize; i++) {
            sum += weights.at(i, j, 0) * grad.at(i, 0, 0);
        }
        gradInput.at(j, 0, 0) = sum;
    }

    return gradInput;
}

void DenseLayer::updateWeights(float learningRate) {
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weights.at(i, j, 0) -= learningRate * gradWeights.at(i, j, 0);
        }
        bias.at(i, 0, 0) -= learningRate * gradBias.at(i, 0, 0);
    }
}