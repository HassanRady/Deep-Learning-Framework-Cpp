#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Layer.hpp"

namespace DeepStorm
{
    namespace Layers
    {
        class Flatten : public Layer
        {
        public:
            Flatten();

            torch::Tensor forward(torch::Tensor &inputTensor) override;

            torch::Tensor backward(torch::Tensor &errorTensor) override;

        private:
            torch::Tensor inputTensor;
        };
    } // namespace Layers

} // namespace DeepStorm
