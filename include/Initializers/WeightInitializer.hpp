#pragma once

#include "torch/torch.h"

namespace DeepStorm
{
    class WeightInitializer
    {
    public:
        virtual void initialize(torch::Tensor &tensor, int fanIn, int fanOut) = 0;
        virtual ~WeightInitializer() = default;
    };
} // namespace DeepStorm
