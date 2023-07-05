#pragma once

#include "torch/torch.h"

#include "WeightInitializer.h"

namespace DeepStorm
{
    namespace Initializers
    {
        class Constant : public WeighInitializer
        {
            Constant(float scalar = 0.1);

            int initialize(torch::Tensor &tensor, int fanIn, int fanOut) override;

        private:
            float scalar;
        };
    } // namespace Initializers

} // namespace DeepStorm
