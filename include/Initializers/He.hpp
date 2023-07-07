#pragma once

#include "torch/torch.h"

#include "WeightInitializer.hpp"

namespace DeepStorm
{
    namespace Initializers
    {
        class He : public WeightInitializer
        {
        public:
            He();
            void initialize(torch::Tensor &tensor, int fanIn, int fanOut) override;
            ~He() = default;
        };
    } // namespace Initializers

} // namespace DeepStorm
