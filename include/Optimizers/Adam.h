#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Optimizer.h"

namespace DeepStorm
{
    namespace Optimizers
    {
        class Adam : public Optimizer
        {
        public:
            Adam(double learningRate = 0.001, double mu = 0.9, double rho = 0.9, double epsilon = 1e-07);

            int update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override;

            ~Adam() = default;

        private:
            double mu, rho, epsilon, k = 0.0;
            torch::Tensor v = torch::zeros({1}, torch::kCUDA);
            torch::Tensor r = torch::zeros({1}, torch::kCUDA);
        };
    }
} // namespace DeepStorm
