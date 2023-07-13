#include "Dropout.hpp"

using namespace DeepStorm::Layers;

Dropout::Dropout(float probability = 0.5)
{
    Dropout::probability = probability;
    Dropout::trainable = false;
    Dropout::initializable = false;
}

void Dropout::forward(torch::Tensor &x)
{
    if (!Dropout::training)
        return;

    auto tensorShape = x.sizes();
    Dropout::mask = torch::rand({tensorShape[tensorShape.size() - 2], tensorShape[tensorShape.size() - 1]}).to(x.device());
    Dropout::mask = torch::less(Dropout::mask, Dropout::probability);

    x = x * Dropout::mask  / Dropout::probability;
}

void Dropout::backward(torch::Tensor &errorTensor)
{
    errorTensor = (errorTensor * Dropout::mask)/Dropout::probability;
}