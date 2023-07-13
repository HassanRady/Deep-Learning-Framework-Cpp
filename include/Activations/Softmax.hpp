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
            void forward(torch::Tensor &x) override;

            void backward(torch::Tensor &y) override;

        private:
            torch::Tensor softmaxOutput;
        };

    } // namespace Activations

} // namespace DeepStorm
