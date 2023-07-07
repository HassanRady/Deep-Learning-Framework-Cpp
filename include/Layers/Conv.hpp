#pragma once

#include "torch/torch.h"
#include "iostream"
#include "string"
#include "vector"

#include "Layer.hpp"
#include "WeightInitializer.hpp"
#include "Optimizer.hpp"

namespace DeepStorm
{
    namespace Layers
    {
        class Conv2d : public Layer
        {
        public:
            Conv2d(int inChannels, int outChannels, torch::ExpandingArray<2> kernelSize, torch::ExpandingArray<2> stride,
                   std::string padding, WeightInitializer *weightInitializer, WeightInitializer *biasInitializer);

            void initialize();

            int getShapeAfterConv(int dimSize, int kernelSize, std::vector<int> pad, int stride);

            std::vector<int> getPadSizeSame(int kernelSize);

            void checkPaddingType(std::string padding);

            /*
             * images shape (BATCHxCHANNELSxHIGHTxWIDTH)
             * */
            torch::Tensor padImagesSame(torch::Tensor &images);

            torch::Tensor removePad(torch::Tensor image);

            torch::Tensor padImages(torch::Tensor &images);

            std::vector<int> getForwardOutputShape(int inputSizeDim1, int inputSizeDim2);

            torch::Tensor convolve(torch::Tensor &slice, torch::Tensor &kernel, torch::Tensor &bias);

            torch::Tensor forward(torch::Tensor &inputTensor) override;

            torch::Tensor backward(torch::Tensor &errorTensor) override;

            Optimizer *optimizer;

        private:
            int batchSize;

            int inChannels;
            int outChannels;

            int kernelSize;
            int kernelSizeDim1;
            int kernelSizeDim2;

            int stride;
            int strideDim1;
            int strideDim2;

            std::string padding;
            std::vector<int> padSizeDim1;
            std::vector<int> padSizeDim2;

            WeightInitializer *weightInitializer;
            WeightInitializer *biasInitializer;
            torch::Tensor weights;
            torch::Tensor bias;
            torch::Tensor gradWeight;
            torch::Tensor gradBias;

            torch::Tensor inputTensor;
            torch::Tensor inputTensorPadded;

            std::vector<int> forwardOutputShape;
        };
    } // namespace Layers

} // namespace DeepStorm
