#pragma once

#include "torch/torch.h"

#include "Layer.hpp"

namespace DeepStorm
{
    namespace Activations
    {
        class SoftMax : public Layer
        {
        public:
            SoftMax();
            torch::Tensor forward(torch::Tensor &x) override;

            torch::Tensor backward(torch::Tensor &y) override;

        private:
            torch::Tensor softmaxOutput;
        };

    } // namespace Activations

} // namespace DeepStorm
