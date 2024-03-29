#pragma once

#include "torch/torch.h"
#include "iostream"

#include "Layer.hpp"
#include "WeightInitializer.hpp"

namespace DeepStorm
{
    namespace Layers
    {
        class Linear : public Layer
        {
        public:
            Linear(int inFeatures, int outFeatures, WeightInitializer *weightInitializer, WeightInitializer *biasInitializer, Optimizer * optimizer);

            void initialize();

            /*
            inputTensor shape: BATCHxFEATURES
            */
            torch::Tensor forward(torch::Tensor &inputTensor) override;

            torch::Tensor backward(torch::Tensor &errorTensor) override;

            Optimizer *optimizer;
        private:
            int inFeatures;
            int outFeatures;

            torch::Tensor weights;
            torch::Tensor gradientWeights;

            WeightInitializer *weightInitializer;
            WeightInitializer *biasInitializer;

            torch::Tensor bias;
            torch::Tensor gradientBias;

            torch::Tensor inputTensor;

            int batchSize;
        };
    }
}
