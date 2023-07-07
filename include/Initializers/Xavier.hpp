#pragma once

#include "torch/torch.h"

#include "WeightInitializer.hpp"

namespace DeepStorm
{
    namespace Initializers
    {
        class Xavier : public WeighInitializer
        {
        public:
            void initialize(torch::Tensor &tensor, int fanIn, int fanOut) override;
        };
    } // namespace Initializers

} // namespace DeepStorm
