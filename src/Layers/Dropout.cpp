#include "Dropout.hpp"

using namespace DeepStorm::Layers;

Dropout::Dropout(float probability = 0.5) : probability(probability)
{
    Dropout::trainable = false;
    Dropout::initializable = false;
}

torch::Tensor Dropout::forward(torch::Tensor &inputTensor) 
{
    if (Dropout::training)
        return inputTensor;

    auto tensorShape = inputTensor.sizes();
    mask = torch::rand({tensorShape[tensorShape.size() - 2], tensorShape[tensorShape.size() - 1]}).to(inputTensor.device());
    return mask * inputTensor;
}

torch::Tensor Dropout::backward(torch::Tensor &errorTensor) 
{
    auto out = errorTensor * mask;
    out = out / probability;
    return out;
}