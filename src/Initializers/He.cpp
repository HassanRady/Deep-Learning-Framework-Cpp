#include "He.hpp"

using namespace DeepStorm::Initializers;

He::He() {}

void He::initialize(torch::Tensor &tensor, int fanIn, int fanOut)
{
    auto sigma = std::sqrt((2.0) / (fanIn));
    torch::nn::init::normal_(tensor, 0, sigma);
}
