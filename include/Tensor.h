#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

class Tensor {
    public:

        // Default constructor
        Tensor() : width(0), height(0), depth(0) {}

        Tensor(int width, int height, int depth);
        float &at(int x, int y, int z);
        float at(int x, int y, int z) const;

        int getWidth() const;
        int getHeight() const;
        int getDepth() const;

    private:
        int width;
        int height;
        int depth;
        std::vector<float> data;
};

#endif