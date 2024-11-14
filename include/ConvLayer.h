#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <iostream>
#include <vector>
#include "Layer.h"

class ConvLayer : public Layer {
    public:
        ConvLayer(int kernelSize, int numFilters);
        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &grad) override;
    private:
        int kernelSize;
        int numFilters;
        std::vector<Tensor> filters;
        
        void initWeights();
};

#endif