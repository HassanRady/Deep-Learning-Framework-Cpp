#include "Pooling.hpp"

using namespace DeepStorm::Layers;

MaxPool2d::MaxPool2d(torch::ExpandingArray<2> kernelSize, torch::ExpandingArray<2> stride)
{
    MaxPool2d::name = "MaxPool2d";
    MaxPool2d::trainable = false;
    MaxPool2d::initializable = false;

    MaxPool2d::kernelSizeDim1 = kernelSize->operator[](0);
    MaxPool2d::kernelSizeDim2 = kernelSize->operator[](1);

    MaxPool2d::strideSizeDim1 = stride->operator[](0);
    MaxPool2d::strideSizeDim2 = stride->operator[](1);
}

int MaxPool2d::getShapeAfterPooling(int dimSize, int kernelSize, int stride)
{
    return (int)1 + (dimSize - kernelSize) / stride;
}

std::vector<int> MaxPool2d::getForwardOutputShape(int inputSizeDim1, int inputSizeDim2)
{
    int outputSizeDim1 = MaxPool2d::getShapeAfterPooling(inputSizeDim1, MaxPool2d::kernelSizeDim1, MaxPool2d::strideSizeDim1);
    int outputSizeDim2 = MaxPool2d::getShapeAfterPooling(inputSizeDim2, MaxPool2d::kernelSizeDim2, MaxPool2d::strideSizeDim2);
    return {MaxPool2d::batchSize, MaxPool2d::outChannels, outputSizeDim1, outputSizeDim2};
}

torch::Tensor MaxPool2d::forward(torch::Tensor &inputTensor)
{
    MaxPool2d::inputTensor = inputTensor;
    batchSize = inputTensor.sizes()[0];
    MaxPool2d::outChannels = inputTensor.sizes()[1];
    int inputSizeDim1 = inputTensor.sizes()[2];
    int inputSizeDim2 = inputTensor.sizes()[3];

    std::vector<int> forwardOutputShape = MaxPool2d::getForwardOutputShape(inputSizeDim1, inputSizeDim2);
    int outputSizeDim1 = forwardOutputShape[2];
    int outputSizeDim2 = forwardOutputShape[3];

    torch::Tensor output = torch::empty({batchSize, outChannels, outputSizeDim1, outputSizeDim2}, torch::kCUDA);

    for (int n = 0; n < batchSize; ++n)
    {
        for (int outChannel = 0; outChannel < outChannels; ++outChannel)
        {
            for (int i = 0; i < outputSizeDim1; ++i)
            {
                for (int j = 0; j < outputSizeDim2; ++j)
                {
                    int startDim1 = i * MaxPool2d::strideSizeDim1;
                    int endDim1 = startDim1 + MaxPool2d::kernelSizeDim1;
                    int startDim2 = j * MaxPool2d::strideSizeDim2;
                    int endDim2 = startDim2 + MaxPool2d::kernelSizeDim2;
                    torch::Tensor slice = inputTensor.index(
                        {n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                         torch::indexing::Slice(startDim2, endDim2)});
                    output.index_put_({n, outChannel, i, j}, torch::max(slice));
                }
            }
        }
    }
    return output;
}

torch::Tensor MaxPool2d::backward(torch::Tensor &errorTensor)
{
    int outputSizeDim1 = errorTensor.sizes()[2];
    int outputSizeDim2 = errorTensor.sizes()[3];
    torch::Tensor gradInput = torch::zeros_like(MaxPool2d::inputTensor);

    for (int n = 0; n < MaxPool2d::batchSize; ++n)
    {
        for (int outChannel = 0; outChannel < MaxPool2d::outChannels; ++outChannel)
        {
            for (int i = 0; i < outputSizeDim1; ++i)
            {
                for (int j = 0; j < outputSizeDim2; ++j)
                {
                    int startDim1 = i * MaxPool2d::strideSizeDim1;
                    int endDim1 = startDim1 + MaxPool2d::kernelSizeDim1;
                    int startDim2 = j * MaxPool2d::strideSizeDim2;
                    int endDim2 = startDim2 + MaxPool2d::kernelSizeDim2;
                    torch::Tensor slice = MaxPool2d::inputTensor.index(
                        {n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                         torch::indexing::Slice(startDim2, endDim2)});

                    auto mask = torch::eq(slice, torch::max(slice));
                    gradInput.index_put_({n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                                          torch::indexing::Slice(startDim2, endDim2)},
                                         gradInput.index({n, outChannel, torch::indexing::Slice(startDim1, endDim1),
                                                          torch::indexing::Slice(startDim2, endDim2)}) +
                                             mask * errorTensor.index({n, outChannel, i, j}));
                }
            }
        }
    }

    return gradInput;
}
