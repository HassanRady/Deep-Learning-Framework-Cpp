#pragma once

#include "torch/torch.h"

#include "WeightInitializer.hpp"

namespace DeepStorm
{
    namespace Initializers
    {
        class Constant : public WeightInitializer
        {
            public:
            Constant(float scalar = 0.1);

            void initialize(torch::Tensor &tensor, int fanIn, int fanOut) override;

        private:
            float scalar;
        };
    } // namespace Initializers

} // namespace DeepStorm
