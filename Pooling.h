#include "torch/torch.h"
#include "iostream"
#include "vector"

#include "Base.h"

class MaxPool2d : public BaseLayer {
public:
    MaxPool2d(torch::ExpandingArray<2> kernelSize, torch::ExpandingArray<2> stride) {
        trainable = false;
        initializable = false;

        kernelSizeDim1 = kernelSize->operator[](0);
        kernelSizeDim2 = kernelSize->operator[](1);

        strideSizeDim1 = stride->operator[](0);
        strideSizeDim2 = stride->operator[](1);
    }

    int getShapeAfterPooling(int dimSize, int kernelSize, int stride) {
        return (int) 1 + (dimSize - kernelSize) / stride;
    }

    std::vector<int> getForwardOutputShape(int inputSizeDim1, int inputSizeDim2) {
        int outputSizeDim1 = getShapeAfterPooling(inputSizeDim1, kernelSizeDim1, strideSizeDim1);
        int outputSizeDim2 = getShapeAfterPooling(inputSizeDim2, kernelSizeDim2, strideSizeDim2);
        return {batchSize, outChannels, outputSizeDim1, outputSizeDim2};
    }

    torch::Tensor forward(torch::Tensor& inputTensor) override{
        this->inputTensor = inputTensor;
        batchSize = inputTensor.sizes()[0];
        outChannels = inputTensor.sizes()[1];
        int inputSizeDim1 = inputTensor.sizes()[2];
        int inputSizeDim2 = inputTensor.sizes()[3];

        std::vector<int> forwardOutputShape = getForwardOutputShape(inputSizeDim1, inputSizeDim2);
        int outputSizeDim1 = forwardOutputShape[2];
        int outputSizeDim2 = forwardOutputShape[3];

        torch::Tensor output = torch::empty({batchSize, outChannels, outputSizeDim1, outputSizeDim2}, torch::kCUDA);

        for (int n = 0; n < batchSize; ++n) {
            for (int outChannel = 0; outChannel < outChannels; ++outChannel) {
                for (int i = 0; i < outputSizeDim1; ++i) {
                    for (int j = 0; j < outputSizeDim2; ++j) {
                        int startDim1 = i * strideSizeDim1;
                        int endDim1 = startDim1 + kernelSizeDim1;
                        int startDim2 = j * strideSizeDim2;
                        int endDim2 = startDim2 + kernelSizeDim2;
                        torch::Tensor slice = inputTensor.index(
                                {n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                                 torch::indexing::Slice(startDim2, endDim2)});
                        output.index_put_({n, outChannel, i, j}, torch::max(slice));
                    }
                }
            }
        }
        std::cout << output.sizes() << "\n";
        return output;
    }

    torch::Tensor backward(torch::Tensor& errorTensor) override{
        int outputSizeDim1 = errorTensor.sizes()[2];
        int outputSizeDim2 = errorTensor.sizes()[3];
        torch::Tensor gradInput = torch::zeros_like(inputTensor);


        for (int n = 0; n < batchSize; ++n) {
            for (int outChannel = 0; outChannel < outChannels; ++outChannel) {
                for (int i = 0; i < outputSizeDim1; ++i) {
                    for (int j = 0; j < outputSizeDim2; ++j) {
                        int startDim1 = i * strideSizeDim1;
                        int endDim1 = startDim1 + kernelSizeDim1;
                        int startDim2 = j * strideSizeDim2;
                        int endDim2 = startDim2 + kernelSizeDim2;
                        torch::Tensor slice = inputTensor.index(
                                {n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                                 torch::indexing::Slice(startDim2, endDim2)});

                        auto mask = torch::eq(slice, torch::max(slice));
                        gradInput.index_put_({n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                                              torch::indexing::Slice(startDim2, endDim2)},
                                             mask * errorTensor.index({n, outChannel, i, j}) +
                                             gradInput.index({n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                                                              torch::indexing::Slice(startDim2, endDim2)}));
                    }
                }
            }
        }


        return gradInput;
    }

private:
    int batchSize;

    int outChannels;

    int kernelSizeDim1;
    int kernelSizeDim2;

    int strideSizeDim1;
    int strideSizeDim2;

    torch::Tensor inputTensor;
};