#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Layer.hpp"

namespace DeepStorm
{
    namespace Activations
    {
        class Sigmoid : public Layer
        {
        public:
            Sigmoid();

            torch::Tensor forward(torch::Tensor &x) override;

            torch::Tensor backward(torch::Tensor &y) override;

            torch::Tensor forwardOutput;
        };
    } // namespace Activations

} // namespace DeepStorm
