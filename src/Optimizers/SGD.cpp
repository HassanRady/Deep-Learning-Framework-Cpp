#include "SGD.hpp"

using namespace DeepStorm::Optimizers;

Sgd::Sgd(double learningRate = 1e-3)
{
    Sgd::learningRate = learningRate;
}

torch::Tensor Sgd::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor)
{
    return weightTensor - Sgd::learningRate * gradientTensor;
}

SgdWithMomentum::SgdWithMomentum(double learningRate, double momentum)
{
    SgdWithMomentum::learningRate = learningRate;
    SgdWithMomentum::momentum = momentum;
}

torch::Tensor SgdWithMomentum::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor)
{
    v = SgdWithMomentum::momentum * v - SgdWithMomentum::learningRate * gradientTensor;
    return weightTensor + v;
}