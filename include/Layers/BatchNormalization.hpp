#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Layer.hpp"
#include "Optimizer.hpp"

namespace DeepStorm
{
    namespace Layers
    {
        class BatchNorm2d : public Layer
        {
        public:
            BatchNorm2d(int numFeatures, Optimizer * optimizer, float eps, float momentum);

            torch::Tensor normalizeTrain(torch::Tensor &tensor);

            torch::Tensor normalizeTest(torch::Tensor &tensor);

            torch::Tensor forward(torch::Tensor &inputTensor) override;

            torch::Tensor backward(torch::Tensor &errorTensor) override;

            Optimizer *optimizer;

        private:
            int batchSize;

            int numFeatures;
            float eps;
            float momentum;

            torch::Tensor weight;
            torch::Tensor bias;

            torch::Tensor mean;
            torch::Tensor variance;

            torch::Tensor inputTensor;
            torch::Tensor inputTensorNormalized;
        };
    } // namespace Layers

} // namespace DeepStorm
