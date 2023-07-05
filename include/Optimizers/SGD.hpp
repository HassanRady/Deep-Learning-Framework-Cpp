#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Optimizer.h"

namespace DeepStorm
{
    namespace Optimizers
    {
        class Sgd : public Optimizer
        {
        public:
            Sgd(double learningRate);

            void update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override;

            ~Sgd() = default;
        };


        class SgdWithMomentum : public Optimizer
        {
        public:
            SgdWithMomentum(double learningRate, double momentum);

            void update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override;

            ~SgdWithMomentum() = 0;

        private:
            double momentum;
            torch::Tensor v = torch::zeros({1}, torch::kCUDA);
        };
    } // namespace Optimizers

} // namespace DeepStorm
