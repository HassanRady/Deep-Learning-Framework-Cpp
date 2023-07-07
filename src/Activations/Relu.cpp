#include "Relu.hpp"

using namespace DeepStorm::Activations;

Relu::Relu() {}

torch::Tensor Relu::forward(torch::Tensor &x)
{
    pos = x.greater_(0);
    return torch::max(x, torch::zeros_like(x, torch::kCUDA));
}

torch::Tensor Relu::backward(torch::Tensor &y)
{
    return pos * y;
}