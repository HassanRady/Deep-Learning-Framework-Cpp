#include "Conv.h"

using namespace DeepStorm::Layers;

Conv2d::Conv2d(int inChannels, int outChannels, torch::ExpandingArray<2> kernelSize, torch::ExpandingArray<2> stride,
       std::string padding,
       WeightInitializer *weightInitializer, WeightInitializer *biasInitializer)
{
    trainable = true;
    initializable = true;

    this->inChannels = inChannels;
    this->outChannels = outChannels;

    kernelSizeDim1 = kernelSize->operator[](0);
    kernelSizeDim2 = kernelSize->operator[](1);

    strideDim1 = stride->operator[](0);
    strideDim2 = stride->operator[](1);

    this->padding = padding;

    this->weightInitializer = weightInitializer;
    this->biasInitializer = biasInitializer;

    initialize();
}

void Conv2d::initialize()
{
    weights = torch::empty({outChannels, inChannels, kernelSizeDim1, kernelSizeDim2}, torch::kCUDA);
    bias = torch::empty({outChannels, 1}, torch::kCUDA);

    weightInitializer->initialize(weights, inChannels * kernelSizeDim1 * kernelSizeDim2,
                                  kernelSizeDim1 * kernelSizeDim2 * outChannels);
    biasInitializer->initialize(bias, outChannels, 1);
}

int Conv2d::getShapeAfterConv(int dimSize, int kernelSize, std::vector<int> pad, int stride)
{
    int startPad = pad[0];
    int endPad = pad[1];
    return (int)1 + (dimSize - kernelSize + startPad + endPad) / stride;
}

std::vector<int> Conv2d::getPadSizeSame(int kernelSize)
{
    int startPad;
    if (kernelSize % 2 == 1)
    {
        startPad = (int)(kernelSize - 1) / 2;
        return {startPad, startPad};
    }
    startPad = (int)kernelSize / 2 - 1;
    int endPad = (int)kernelSize / 2;
    return {startPad, endPad};
}

void Conv2d::checkPaddingType(std::string padding)
{
    if (padding == "same")
    {
        padSizeDim1 = Conv2d::getPadSizeSame(kernelSizeDim1);
        padSizeDim2 = Conv2d::getPadSizeSame(kernelSizeDim2);
    }
    else if (padding == "valid")
    {
        padSizeDim1 = {0, 0};
        padSizeDim2 = {0, 0};
    }
    //        else if (isdigit(padding)) {
    //            padSizeDim1 = {padding, padding};
    //            padSizeDim2 = {padding, padding};
    //        }
}

/*
 * images shape (BATCHxCHANNELSxHIGHTxWIDTH)
 * */
torch::Tensor Conv2d::padImagesSame(torch::Tensor &images)
{
    int startPadDim1 = padSizeDim1[0];
    int endPadDim1 = padSizeDim1[1];
    int startPadDim2 = padSizeDim2[0];
    int endPadDim2 = padSizeDim2[1];

    std::vector<int64_t> padding = {startPadDim2, endPadDim2, startPadDim1, endPadDim1, 0, 0};
    torch::nn::functional::PadFuncOptions options(padding);

    return torch::nn::functional::pad(images, options);
}

/*
 * image shape CHANNELSxHIGHTxWIDTH
 **/
torch::Tensor Conv2d::removePad(torch::Tensor image)
{
    int startPadDim1 = padSizeDim1[0];
    int endPadDim1 = padSizeDim1[1];
    int startPadDim2 = padSizeDim2[0];
    int endPadDim2 = padSizeDim2[1];
    auto imageSize = image.sizes();
    return image.index({torch::indexing::Slice(), torch::indexing::Slice(startPadDim1, imageSize[1] - endPadDim1),
                        torch::indexing::Slice(startPadDim2, imageSize[2] - endPadDim2)});
}

torch::Tensor Conv2d::padImages(torch::Tensor &images)
{
    checkPaddingType(padding);
    if (padding == "same")
        return padImagesSame(images);
}

std::vector<int> Conv2d::getForwardOutputShape(int inputSizeDim1, int inputSizeDim2)
{
    int outputSizeDim1 = Conv2d::getShapeAfterConv(inputSizeDim1, kernelSizeDim1, padSizeDim1, strideDim1);
    int outputSizeDim2 = Conv2d::getShapeAfterConv(inputSizeDim2, kernelSizeDim2, padSizeDim2, strideDim2);
    return {batchSize, outChannels, outputSizeDim1, outputSizeDim2};
}

torch::Tensor Conv2d::convolve(torch::Tensor &slice, torch::Tensor &kernel, torch::Tensor &bias)
{
    return torch::sum(slice * kernel) + bias;
}

torch::Tensor Conv2d::forward(torch::Tensor &inputTensor) override
{
    this->inputTensor = inputTensor;
    batchSize = inputTensor.sizes()[0];
    auto inputSizeDim1 = inputTensor.sizes()[2];
    auto inputSizeDim2 = inputTensor.sizes()[3];
    inputTensorPadded = Conv2d::padImages(inputTensor);

    forwardOutputShape = Conv2d::getForwardOutputShape(inputSizeDim1, inputSizeDim2);
    int outputSizeDim1 = forwardOutputShape[2];
    int outputSizeDim2 = forwardOutputShape[3];
    torch::Tensor forwardOutput = torch::empty({batchSize, outChannels, outputSizeDim1, outputSizeDim2}, torch::kCUDA);

    for (int n = 0; n < batchSize; ++n)
    {
        for (int outChannel = 0; outChannel < outChannels; ++outChannel)
        {
            torch::Tensor kernel = weights.index({outChannel});
            torch::Tensor bias = this->bias.index({outChannel});

            for (int i = 0; i < outputSizeDim1; ++i)
            {
                for (int j = 0; j < outputSizeDim2; ++j)
                {
                    int startDim1 = i * strideDim1;
                    int endDim1 = startDim1 + kernelSizeDim1;
                    int startDim2 = j * strideDim2;
                    int endDim2 = startDim2 + kernelSizeDim2;
                    torch::Tensor slice = inputTensorPadded.index(
                        {n, torch::indexing::Slice(), torch::indexing::Slice(startDim1, endDim1),
                         torch::indexing::Slice(startDim2, endDim2)});

                    forwardOutput.index_put_({n, outChannel, i, j}, convolve(slice, kernel, bias).to(torch::kCUDA));
                }
            }
        }
    }

    return forwardOutput;
}

torch::Tensor Conv2d::backward(torch::Tensor &errorTensor) override
{
    int outputSizeDim1 = errorTensor.sizes()[2];
    int outputSizeDim2 = errorTensor.sizes()[3];

    torch::Tensor backwardOutput = torch::empty_like(inputTensor);
    torch::Tensor gradInput = torch::empty_like(inputTensorPadded);
    gradWeight = torch::empty_like(weights);
    gradBias = torch::empty({outChannels, 1, 1, 1}, torch::kCUDA);

    for (int n = 0; n < batchSize; ++n)
    {
        for (int outChannel = 0; outChannel < outChannels; ++outChannel)
        {
            for (int i = 0; i < outputSizeDim1; ++i)
            {
                for (int j = 0; j < outputSizeDim2; ++j)
                {
                    int startDim1 = i * strideDim1;
                    int endDim1 = startDim1 + kernelSizeDim1;
                    int startDim2 = j * strideDim2;
                    int endDim2 = startDim2 + kernelSizeDim2;
                    torch::Tensor slice = inputTensorPadded.index(
                        {n, torch::indexing::Slice(), torch::indexing::Slice(startDim1, endDim1),
                         torch::indexing::Slice(startDim2, endDim2)});

                    gradWeight.index_put_({outChannel}, gradWeight.index({outChannel}) +
                                                            slice * errorTensor.index({n, outChannel, i, j}));
                    gradBias.index_put_({outChannel},
                                        gradBias.index({outChannel}) + errorTensor.index({n, outChannel, i, j}));
                    gradInput.index_put_({n, torch::indexing::Slice(), torch::indexing::Slice(startDim1, endDim1),
                                          torch::indexing::Slice(startDim2, endDim2)},
                                         errorTensor.index({n, outChannel, i, j}) * weights.index({outChannel}) +
                                             gradInput.index({n, torch::indexing::Slice(),
                                                              torch::indexing::Slice(startDim1, endDim1),
                                                              torch::indexing::Slice(startDim2, endDim2)}));
                }
            }
        }
        backwardOutput.index_put_({n}, Conv2d::removePad(gradInput.index({n})));
    }
    optimizer->update(weights, gradWeight);
    return backwardOutput;
}