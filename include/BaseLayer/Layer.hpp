#pragma once

#include "torch/torch.h"
#include "Optimizer.hpp"

namespace DeepStorm
{
    class Layer
    {
    public:
        virtual void forward(torch::Tensor &tensor) = 0;

        virtual void backward(torch::Tensor &tensor) = 0;

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