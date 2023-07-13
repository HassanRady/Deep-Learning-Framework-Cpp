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

            void forward(torch::Tensor &inputTensor) override;

            void backward(torch::Tensor &errorTensor) override;

        private:
            torch::Tensor inputTensor;
        };
    } // namespace Layers

} // namespace DeepStorm
