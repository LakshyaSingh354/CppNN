#ifndef CNNMODEL_H
#define CNNMODEL_H

#include "ConvLayer.h"
#include "DenseLayer.h"
#include "Tensor.h"
#include "MaxPool.h"
#include <vector>

class CNNModel {
    public:
        CNNModel();
        Tensor forward(const Tensor &input);
        void backward(const Tensor &grad);
        void updateWeights(float learningRate);

    private:
        std::vector<Layer*> layers;
};

#endif