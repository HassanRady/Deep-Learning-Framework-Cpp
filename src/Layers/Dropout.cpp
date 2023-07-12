#include "Dropout.hpp"

using namespace DeepStorm::Layers;

Dropout::Dropout(float probability = 0.5)
{
    Dropout::probability = probability;
    Dropout::trainable = false;
    Dropout::initializable = false;
}

torch::Tensor Dropout::forward(torch::Tensor &inputTensor)
{
    if (!Dropout::training)
        return inputTensor;

    auto tensorShape = inputTensor.sizes();
    Dropout::mask = torch::rand({tensorShape[tensorShape.size() - 2], tensorShape[tensorShape.size() - 1]}).to(inputTensor.device());
    Dropout::mask = torch::less(Dropout::mask, Dropout::probability);

    return inputTensor * Dropout::mask  / Dropout::probability;
}

torch::Tensor Dropout::backward(torch::Tensor &errorTensor)
{
    auto out = errorTensor * Dropout::mask;
    out = out / Dropout::probability;
    return out;
}