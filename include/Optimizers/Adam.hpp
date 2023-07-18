#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Optimizer.hpp"

namespace DeepStorm
{
    namespace Optimizers
    {
        class Adam : public Optimizer
        {
        public:
            Adam(float learningRate, float mu, float rho, float epsilon);

            torch::Tensor update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override;

            ~Adam() = default;

        private:
            float mu, rho, epsilon;
            float k = 0.0;
            torch::Tensor v = torch::zeros({1}, torch::kCUDA);
            torch::Tensor r = torch::zeros({1}, torch::kCUDA);
        };
    }
} // namespace DeepStorm
