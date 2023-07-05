#include "torch/torch.h"

using namespace DeepStorm::Optimizers;

Sgd::Sgd() {}

void Sgd::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override
{
    weightTensor = weightTensor - learningRate * gradientTensor;
}

SgdWithMomentum::SgdWithMomentum(double learningRate, double momentum) : Optimizer(learningRate), momentum(momentum) {}

void SgdWithMomentum::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override
{
    v = momentum * v - learningRate * gradientTensor;
    weightTensor = weightTensor + v;
}