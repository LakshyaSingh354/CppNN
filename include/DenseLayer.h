#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "Layer.h"
#include "Tensor.h"
#include <vector>

class DenseLayer : public Layer {
    public:
        DenseLayer(int inputSize, int outputSize);

        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &grad) override;
        void updateWeights(float learningRate);

    private:
        int inputSize;
        int outputSize;
        Tensor weights;
        Tensor bias;
        Tensor inputCache;
        Tensor gradWeights;
        Tensor gradBias;
};

#endif