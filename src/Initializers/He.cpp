#include "He.hpp"

using namespace DeepStorm::Initializers;

He::He() {}

void He::initialize(torch::Tensor &tensor, int fanIn, int fanOut) override
{
    auto sigma = std::sqrt((float)(2) / (fanIn + fanOut));
    torch::nn::init::normal_(tensor, 0, sigma);
}