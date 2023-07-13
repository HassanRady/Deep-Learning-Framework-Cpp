#include "Relu.hpp"

using namespace DeepStorm::Activations;

ReLU::ReLU() {}

void  ReLU::forward(torch::Tensor &x)
{
    ReLU::pos = torch::greater(x, 0);
    x = torch::max(x, torch::zeros_like(x, torch::kCUDA));
}

void ReLU::backward(torch::Tensor &y)
{
    y = ReLU::pos * y;
}