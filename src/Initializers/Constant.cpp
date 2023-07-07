#include "Constant.hpp"

using namespace DeepStorm::Initializers;

Constant::Constant(float scalar = 0.1) : scalar(scalar) {}

int Constant::initialize(torch::Tensor &tensor, int fanIn, int fanOut)
{
    tensor.fill_(scalar);
    return 0;
}