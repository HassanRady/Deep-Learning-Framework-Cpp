#include "Constant.hpp"

using namespace DeepStorm::Initializers;

Constant::Constant(float scalar = 0.1) {
    Constant::scalar = scalar;
}

void Constant::initialize(torch::Tensor &tensor, int fanIn, int fanOut)
{
    tensor.fill_(Constant::scalar);
}