#include <iostream>
#include <cstdlib>
#include <cmath>
#include "ConvLayer.h"
#include "/opt/homebrew/opt/libomp/include/omp.h"

using namespace std;

ConvLayer::ConvLayer(int kernelSize, int numFilters)
    : kernelSize(kernelSize), numFilters(numFilters) {
    initWeights();
}

void ConvLayer::initWeights(){
    filters.resize(numFilters, Tensor(kernelSize, kernelSize, 1));
    srand(time(0));
    for(auto &filter: filters){
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                filter.at(i, j, 0) = static_cast<float>(rand()) / RAND_MAX;
            }
            
        }
        
    }
}

Tensor ConvLayer::forward(const Tensor& input) {
    int width = input.getWidth();
    int height = input.getHeight();
    int depth = input.getDepth();
    int filterSize = filters[0].getWidth();

    int outputWidth = width - filterSize + 1;
    int outputHeight = height - filterSize + 1;
    Tensor output(outputWidth, outputHeight, numFilters);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outputWidth; ++i) {
        for (int j = 0; j < outputHeight; ++j) {
            for (int f = 0; f < numFilters; ++f) {
                float sum = 0.0f;
                for (int ki = 0; ki < filterSize; ++ki) {
                    for (int kj = 0; kj < filterSize; ++kj) {
                        sum += input.at(i + ki, j + kj, 0) * filters[f].at(ki, kj, 0);
                    }
                }
                output.at(i, j, f) = sum;
            }
        }
    }

    this->input = input;
    return output;
}

Tensor ConvLayer::backward(const Tensor &grad){
    
    int width = input.getWidth();
    int height = input.getHeight();

    Tensor gradInput(input.getWidth(), input.getHeight(), 1);

    std::vector<Tensor> gradFilters(numFilters, Tensor(kernelSize, kernelSize, 1));

    for(int f = 0; f < numFilters; f++){
        for(int i = 0; i < width; i++){
            for(int j = 0; j < height; j++){
                float grad_val = grad.at(i, j, f);

                for(int ki = 0; ki < kernelSize; ki++){
                    for(int kj = 0; kj < kernelSize; kj++){
                        gradFilters[f].at(ki, kj, 0) += input.at(i + ki, j + kj, 0) * grad_val;
                    }
                }

                for(int ki = 0; ki < kernelSize; ki++){
                    for(int kj = 0; kj < kernelSize; kj++){
                        gradInput.at(i + ki, j + kj, 0) += filters[f].at(ki, kj, 0) * grad_val;
                    }
                }
            }
        }
    }
    return gradInput;
}