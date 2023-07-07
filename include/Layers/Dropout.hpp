#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Layer.hpp"

namespace DeepStorm
{
    namespace Layers
    {
        class Dropout : public Layer
        {
        public:
            Dropout(float probability = 0.5);

            torch::Tensor forward(torch::Tensor &inputTensor) override;

            torch::Tensor backward(torch::Tensor &errorTensor) override;

        private:
            float probability;
            bool testingPhase = false;
            torch::Tensor mask;
        };
    } // namespace Layers

}