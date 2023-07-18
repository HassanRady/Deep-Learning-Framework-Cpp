#pragma once

#include "torch/torch.h"
#include "iostream"

namespace DeepStorm
{
    class Optimizer
    {
    public:
        virtual torch::Tensor update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) = 0;

        virtual ~Optimizer() = default;

    public:
        double learningRate;
    };
} // namespace DeepStorm
