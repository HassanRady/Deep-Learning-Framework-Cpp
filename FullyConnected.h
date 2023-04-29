

#include "Base.h"
#include "Optimizer.h"

#include "iostream"
#include <torch/torch.h>

using namespace torch::indexing;


class Linear: public  BaseLayer{
public: Linear(int inFeatures, int outFeatures) : inputSize(inFeatures), outputSize(outFeatures){
        trainable = true;
        initializable = true;

        weights = torch::randn({inFeatures+1, outFeatures}, torch::kCUDA);
        bias = torch::ones({outFeatures, 1}, torch::kCUDA);
    }

    torch::Tensor forward(torch::Tensor & inputTensor) {
        auto inputDims = inputTensor.sizes();
        batchSize = inputDims[0];

        this->inputTensor = torch::cat({inputTensor, torch::ones({batchSize, 1}, torch::kCUDA)}, -1);
        return torch::matmul(this->inputTensor, weights);
 
        // return torch::matmul(inputTensor, weights) + bias;
}

torch::Tensor backward(torch::Tensor & errorTensor) {

    gradientWeights = torch::matmul(inputTensor.transpose(1, 0), errorTensor);


    weights = optimizer->update(weights, gradientWeights);

    auto out = torch::matmul(errorTensor, weights.transpose(1, 0)).index({Slice(), Slice(0, inputSize)});

    return out;
}

private:
    int inputSize;
    int outputSize;

    torch::Tensor weights;
    torch::Tensor gradientWeights;

    torch::Tensor bias;
    torch::Tensor gradientBias;

    torch::Tensor inputTensor;

    int batchSize;

public:
    Optimizer * optimizer;
};

