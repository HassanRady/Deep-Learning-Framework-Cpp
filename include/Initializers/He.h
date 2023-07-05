#pragma once

#include "torch/torch.h"

#include "WeightInitializer.h"

namespace DeepStorm
{
    namespace Initializers
    {
        class He : public WeightInitializer
        {
        public:
            int initialize(torch::Tensor &tensor, int fanIn, int fanOut) override;
        };
    } // namespace Initializers

} // namespace DeepStorm
