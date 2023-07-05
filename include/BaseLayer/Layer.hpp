#pragma once

#include "torch/torch.h"
#include "Optimizer.h"

namespace DeepStorm
{
    class Layer
    {
    public:
        virtual torch::Tensor forward(torch::Tensor &tensor) = 0;

        virtual torch::Tensor backward(torch::Tensor &tensor) = 0;

        virtual ~Layer() = default;

        bool trainable;
        bool initializable;
        bool training;
        Optimizer *optimizer;
    };
}