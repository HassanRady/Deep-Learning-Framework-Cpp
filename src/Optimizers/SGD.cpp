#include "torch/torch.hpp"

using namespace DeepStorm::Optimizers;

Sgd::Sgd() {}

void Sgd::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) 
{
    weightTensor = weightTensor - learningRate * gradientTensor;
}

SgdWithMomentum::SgdWithMomentum(double learningRate, double momentum) : Optimizer(learningRate), momentum(momentum) {}

void SgdWithMomentum::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) 
{
    v = momentum * v - learningRate * gradientTensor;
    weightTensor = weightTensor + v;
}