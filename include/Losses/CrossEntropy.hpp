#pragma once

#include "torch/torch.h"

#include "Loss.hpp"

namespace DeepStorm
{
    namespace Losses
    {
        class CrossEntropyLoss : public Loss
        {
            public:
            CrossEntropyLoss(float epsilon);

            float forward(torch::Tensor &y_hat, torch::Tensor &y) override;

            torch::Tensor backward(torch::Tensor &grad_y) override;

            ~CrossEntropyLoss() = default;

        private:
            torch::Tensor predictions;
            float epsilon;
        };
    } // namespace Losses

} // namespace DeepStorm
