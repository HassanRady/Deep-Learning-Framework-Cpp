#pragma once

#include "torch/torch.h"

namespace DeepStorm
{
    class WeighInitializer
    {
    public:
        virtual int initialize(torch::Tensor &tensor, int fanIn, int fanOut) = 0;
    };
} // namespace DeepStorm
