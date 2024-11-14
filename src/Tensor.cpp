#include "Tensor.h"

Tensor::Tensor(int width, int height, int depth)
    : width(width), height(height), depth(depth), data(width * height * depth, 0.0f) {}

float &Tensor::at(int x, int y, int z) {
    return data[z * width * height + y * width + x];
}

float Tensor::at(int x, int y, int z) const {
    return data[z * width * height + y * width + x];
}

int Tensor::getWidth() const {
    return width;
}

int Tensor::getHeight() const {
    return height;
}

int Tensor::getDepth() const {
    return depth;
}