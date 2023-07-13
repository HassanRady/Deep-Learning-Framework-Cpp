#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Layer.hpp"

namespace DeepStorm
{
    namespace Activations
    {
        class ReLU : public Layer
        {
        public:
            ReLU();

            void forward(torch::Tensor &x) override;

            void backward(torch::Tensor &y) override;

        private:
            torch::Tensor pos;
        };

    } // namespace Activation

} // namespace DeepStorm
