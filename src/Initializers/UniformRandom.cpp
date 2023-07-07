#include "UniformRandom.hpp"

using namespace DeepStorm::Initializers;

UniformRandom::UniformRandom() {}

void UniformRandom::initialize(torch::Tensor &tensor, int fanIn, int fanOut)
{
    tensor.fill_(torch::rand_like(tensor));
}