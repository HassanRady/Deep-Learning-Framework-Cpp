#include "Relu.hpp"

using namespace DeepStorm::Activations;

ReLU::ReLU() {
    ReLU::name = "ReLU";
    ReLU::trainable = false;
    ReLU::initializable = false;
    ReLU::training = false;
}

torch::Tensor ReLU::forward(torch::Tensor &x)
{
    ReLU::positive = (x > 0).to(torch::kFloat32);
    return x * ReLU::positive;
}

torch::Tensor ReLU::backward(torch::Tensor &y)
{
    return y * ReLU::positive;
}