#pragma once

#include "torch/torch.h"
#include "iostream"
#include "vector"

#include "Layer.hpp"

namespace DeepStorm
{
    namespace Layers
    {
        class MaxPool2d : public Layer
        {
        public:
            MaxPool2d(torch::ExpandingArray<2> kernelSize, torch::ExpandingArray<2> stride);

            int getShapeAfterPooling(int dimSize, int kernelSize, int stride);

            std::vector<int> getForwardOutputShape(int inputSizeDim1, int inputSizeDim2);

            torch::Tensor forward(torch::Tensor &inputTensor) override;

            torch::Tensor backward(torch::Tensor &errorTensor) override;

        private:
            int batchSize;

            int outChannels;

            int kernelSizeDim1;
            int kernelSizeDim2;

            int strideSizeDim1;
            int strideSizeDim2;

            torch::Tensor inputTensor;
        };

    } // namespace Layers

} // namespace DeepStorm
