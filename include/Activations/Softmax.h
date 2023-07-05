#pragma once

#include "torch/torch.h"

#include "Layer.h"

namespace DeepStorm
{
    namespace Activations
    {
        class SoftMax : public Layer
        {
        public:
            torch::Tensor forward(torch::Tensor &x) override;

            torch::Tensor backward(torch::Tensor &y) override;

        private:
            torch::Tensor softmaxOutput;
        };

    } // namespace Activations

} // namespace DeepStorm
