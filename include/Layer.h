#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include "Tensor.h"

class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor forward(const Tensor &input) = 0;
        virtual Tensor backward(const Tensor &grad) = 0;
    protected:
        Tensor input;
};

#endif