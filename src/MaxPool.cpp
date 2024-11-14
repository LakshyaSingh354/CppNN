#include "MaxPool.h"
#include "/opt/homebrew/opt/libomp/include/omp.h"
#include <algorithm>

using namespace std;

MaxPool::MaxPool(int poolSize, int stride)
    : poolSize(poolSize), stride(stride) {}

Tensor MaxPool::forward(const Tensor &input) {
    int width = (input.getWidth() - poolSize) / stride + 1;
    int height = (input.getHeight() - poolSize) / stride + 1;
    int depth = input.getDepth();

    Tensor output(width, height, depth);

    #pragma omp parallel for collapse(3)
    for(int d = 0; d < depth; d++){
        for(int i = 0; i < width; i++){
            for(int j = 0; j < height; j++){
                float maxVal = -numeric_limits<float>::infinity();

                for(int pi = 0; pi < poolSize; pi++){
                    for(int pj = 0; pj < poolSize; pj++){
                        int x = i * stride + pi;
                        int y = j * stride + pj;
                        maxVal = max(maxVal, input.at(x, y, d));
                    }
                }
                output.at(i, j, d) = maxVal;
            }
        }
    }

    this->input = input;
    return output;
}

Tensor MaxPool::backward(const Tensor &grad){
    int width = (grad.getWidth() - 1) * stride + poolSize;
    int height = (grad.getHeight() - 1) * stride + poolSize;
    int depth = grad.getDepth();

    Tensor grad_input(width, height, depth);

    #pragma omp parallel for collapse(3)
    for(int d = 0; d < depth; d++){
        for(int i = 0; i < grad.getWidth(); i++){
            for(int j = 0; j < grad.getHeight(); j++){
                float maxVal = -numeric_limits<float>::infinity();
                int maxI = -1, maxJ = -1;

                for(int pi = 0; pi < poolSize; pi++){
                    for(int pj = 0; pj < poolSize; pj++){
                        int x = i * stride + pi;
                        int y = j * stride + pj;
                        if(input.at(x, y, d) > maxVal){
                            maxVal = input.at(x, y, d);
                            maxI = x;
                            maxJ = y;
                        }
                    }
                }
                grad_input.at(maxI, maxJ, d) = grad.at(i, j, d);
            }
        }
    }

    return grad_input;
}