#pragma once

#include "torch/torch.h"

#include "WeightInitializer.hpp"

namespace DeepStorm
{
    namespace Initializers
    {
        class UniformRandom : public WeightInitializer
        {
        public:
            void initialize(torch::Tensor &tensor, int fanIn, int fanOut) override;
        };
    } // namespace Initializers

} // namespace DeepStorm
