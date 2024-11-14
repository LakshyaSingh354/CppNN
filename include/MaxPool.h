#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "Tensor.h"
#include "Layer.h"

class MaxPool : public Layer {
    public:
        MaxPool(int poolSize, int stride);

        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &grad) override;

    private:
        int poolSize;
        int stride;
};

#endif