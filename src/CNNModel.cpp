#include "CNNModel.h"
#include "ConvLayer.h"
#include "DenseLayer.h"
#include "MaxPool.h"

CNNModel::CNNModel() {
    layers.push_back(new ConvLayer(3, 8));
    layers.push_back(new MaxPool(2, 0));
    layers.push_back(new ConvLayer(3, 16));
    layers.push_back(new MaxPool(2, 0));
    layers.push_back(new DenseLayer(2, 3));
}

Tensor CNNModel::forward(const Tensor &input) {
    Tensor output = input;
    for (auto layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void CNNModel::backward(const Tensor &grad) {
    Tensor output = grad;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        output = (*it)->backward(output);
    }
}

void CNNModel::updateWeights(float learningRate) {
    for (auto layer : layers) {
        layer->updateWeights(learningRate);
    }
}