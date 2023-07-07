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
            void initialize(torch::Tensor &tensor, int fanIn, int fanOut) override;
            ~He(){}
        };
    } // namespace Initializers

} // namespace DeepStorm
