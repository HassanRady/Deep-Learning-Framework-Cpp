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
            Dropout(float probability);

            void forward(torch::Tensor &x) override;

            void backward(torch::Tensor &errorTensor) override;

        private:
            float probability;
            bool testingPhase = false;
            torch::Tensor mask;
        };
    } // namespace Layers

}