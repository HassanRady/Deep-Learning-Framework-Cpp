#pragma once

#include "torch/torch.h"
#include "Optimizer.hpp"

namespace DeepStorm
{
    class Layer
    {
    public:
        virtual torch::Tensor forward(torch::Tensor &tensor) = 0;

        virtual torch::Tensor backward(torch::Tensor &tensor) = 0;

        void train() {
            training = true;
        }

        void eval() {
            training = false;
        }

        virtual ~Layer() = default;

        bool trainable;
        bool initializable;
        bool training;
        Optimizer *optimizer;
    };
}