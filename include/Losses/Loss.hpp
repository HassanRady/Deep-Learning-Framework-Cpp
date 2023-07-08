#pragma once

#include "torch/torch.h"

namespace DeepStorm
{
    class Loss
    {
    public:
        Loss(){};
        virtual float forward(torch::Tensor &y_hat, torch::Tensor &y) = 0;

        virtual torch::Tensor backward(torch::Tensor &grad_y) = 0;

        virtual ~Loss() = default;
    };
} // namespace DeepStorm
