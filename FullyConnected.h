

#include "Base.h"
#include "Optimizer.h"

#include "iostream"
#include <torch/torch.h>


class Linear: public  BaseLayer{
public: Linear(int inFeatures, int outFeatures) : inputSize(inFeatures), outputSize(outFeatures){
        trainable = true;
        initializable = true;

        weights = torch::randn({inFeatures, outFeatures}, torch::kCUDA);
        bias = torch::ones(inFeatures);
    }

    torch::Tensor forward(const torch::Tensor & inputTensor) {
        // return torch::multiply(inputTensor, weights);
        return torch::matmul(inputTensor, weights);
}

private:
    int inputSize;
    int outputSize;
    torch::Tensor weights;
    torch::Tensor bias;
public:
    Optimizer optimizer;
};

