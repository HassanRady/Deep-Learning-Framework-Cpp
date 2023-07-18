#include "Sigmoid.hpp"

using namespace DeepStorm::Activations;

Sigmoid::Sigmoid(){}

torch::Tensor Sigmoid::forward(torch::Tensor & x) {
    Sigmoid::forwardOutput = (1)/(1 + torch::exp(x));
    return Sigmoid::forwardOutput;
}

torch::Tensor Sigmoid::backward(torch::Tensor & y) {
    return Sigmoid::forwardOutput * (1 - Sigmoid::forwardOutput);
}