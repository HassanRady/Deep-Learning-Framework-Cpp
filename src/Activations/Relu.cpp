#include "Relu.hpp"

using namespace DeepStorm::Activations;

ReLU::ReLU() {}

torch::Tensor ReLU::forward(torch::Tensor &x)
{
    ReLU::pos = x.greater_(0);
    return torch::max(x, torch::zeros_like(x, torch::kCUDA));
}

torch::Tensor ReLU::backward(torch::Tensor &y)
{
    return ReLU::pos * y;
}