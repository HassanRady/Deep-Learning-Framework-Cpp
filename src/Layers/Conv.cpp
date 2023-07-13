#include "Conv.hpp"

using namespace DeepStorm::Layers;

Conv2d::Conv2d(int inChannels, int outChannels, torch::ExpandingArray<2> kernelSize, torch::ExpandingArray<2> stride,
       std::string padding,
       WeightInitializer *weightInitializer, WeightInitializer *biasInitializer, Optimizer * optimizer)
{
    Conv2d::trainable = true;
    Conv2d::initializable = true;

    Conv2d::inChannels = inChannels;
    Conv2d::outChannels = outChannels;

    Conv2d::kernelSizeDim1 = kernelSize->operator[](0);
    Conv2d::kernelSizeDim2 = kernelSize->operator[](1);

    Conv2d::strideDim1 = stride->operator[](0);
    Conv2d::strideDim2 = stride->operator[](1);

    Conv2d::padding = padding;

    Conv2d::weightInitializer = weightInitializer;
    Conv2d::biasInitializer = biasInitializer;

    Conv2d::optimizer = optimizer;

    Conv2d::initialize();
}

void Conv2d::initialize()
{
    Conv2d::weights = torch::empty({outChannels, inChannels, kernelSizeDim1, kernelSizeDim2}, torch::kCUDA);
    Conv2d::bias = torch::empty({outChannels, 1}, torch::kCUDA);

    Conv2d::weightInitializer->initialize(Conv2d::weights, inChannels * kernelSizeDim1 * kernelSizeDim2,
                                  kernelSizeDim1 * kernelSizeDim2 * outChannels);
    Conv2d::biasInitializer->initialize(bias, outChannels, 1);
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
        Conv2d::padSizeDim1 = Conv2d::getPadSizeSame(Conv2d::kernelSizeDim1);
        Conv2d::padSizeDim2 = Conv2d::getPadSizeSame(Conv2d::kernelSizeDim2);
    }
    else if (padding == "valid")
    {
        Conv2d::padSizeDim1 = {0, 0};
        Conv2d::padSizeDim2 = {0, 0};
    }
    //        else if (isdigit(padding)) {
    //            Conv2d::padSizeDim1 = {padding, padding};
    //            Conv2d::padSizeDim2 = {padding, padding};
    //        }
}

/*
 * images shape (BATCHxCHANNELSxHIGHTxWIDTH)
 * */
torch::Tensor Conv2d::padImagesSame(torch::Tensor &images)
{
    int startPadDim1 = Conv2d::padSizeDim1[0];
    int endPadDim1 = Conv2d::padSizeDim1[1];
    int startPadDim2 = Conv2d::padSizeDim2[0];
    int endPadDim2 = Conv2d::padSizeDim2[1];

    std::vector<int64_t> padding = {startPadDim2, endPadDim2, startPadDim1, endPadDim1, 0, 0};
    torch::nn::functional::PadFuncOptions options(padding);

    return torch::nn::functional::pad(images, options);
}

/*
 * image shape CHANNELSxHIGHTxWIDTH
 **/
torch::Tensor Conv2d::removePad(torch::Tensor image)
{
    int startPadDim1 = Conv2d::padSizeDim1[0];
    int endPadDim1 = Conv2d::padSizeDim1[1];
    int startPadDim2 = Conv2d::padSizeDim2[0];
    int endPadDim2 = Conv2d::padSizeDim2[1];
    auto imageSize = image.sizes();
    return image.index({torch::indexing::Slice(), torch::indexing::Slice(startPadDim1, imageSize[1] - endPadDim1),
                        torch::indexing::Slice(startPadDim2, imageSize[2] - endPadDim2)});
}

torch::Tensor Conv2d::padImages(torch::Tensor &images)
{
    Conv2d::checkPaddingType(Conv2d::padding);
    if (Conv2d::padding == "same")
        return Conv2d::padImagesSame(images);
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

void Conv2d::forward(torch::Tensor &x) 
{
    Conv2d::inputTensor = x;
    Conv2d::batchSize = x.sizes()[0];
    auto inputSizeDim1 = x.sizes()[2];
    auto inputSizeDim2 = x.sizes()[3];
    inputTensorPadded = Conv2d::padImages(x);

    Conv2d::forwardOutputShape = Conv2d::getForwardOutputShape(inputSizeDim1, inputSizeDim2);
    int outputSizeDim1 = Conv2d::forwardOutputShape[2];
    int outputSizeDim2 = Conv2d::forwardOutputShape[3];
    torch::Tensor forwardOutput = torch::empty({Conv2d::batchSize, outChannels, outputSizeDim1, outputSizeDim2}, torch::kCUDA);

    for (int n = 0; n < Conv2d::batchSize; ++n)
    {
        for (int outChannel = 0; outChannel < outChannels; ++outChannel)
        {
            torch::Tensor kernel = Conv2d::weights.index({outChannel});
            torch::Tensor bias = Conv2d::bias.index({outChannel});

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

                    forwardOutput.index_put_({n, outChannel, i, j}, Conv2d::convolve(slice, kernel, bias).to(torch::kCUDA));
                }
            }
        }
    }

    x = forwardOutput;
}

void Conv2d::backward(torch::Tensor &errorTensor) 
{
    int outputSizeDim1 = errorTensor.sizes()[2];
    int outputSizeDim2 = errorTensor.sizes()[3];

    torch::Tensor backwardOutput = torch::empty_like(Conv2d::inputTensor);
    torch::Tensor gradInput = torch::empty_like(Conv2d::inputTensorPadded);
    Conv2d::gradWeight = torch::empty_like(Conv2d::weights);
    Conv2d::gradBias = torch::empty({Conv2d::outChannels, 1, 1, 1}, torch::kCUDA);

    for (int n = 0; n < Conv2d::batchSize; ++n)
    {
        for (int outChannel = 0; outChannel < Conv2d::outChannels; ++outChannel)
        {
            for (int i = 0; i < outputSizeDim1; ++i)
            {
                for (int j = 0; j < outputSizeDim2; ++j)
                {
                    int startDim1 = i * Conv2d::strideDim1;
                    int endDim1 = startDim1 + Conv2d::kernelSizeDim1;
                    int startDim2 = j * Conv2d::strideDim2;
                    int endDim2 = startDim2 + Conv2d::kernelSizeDim2;
                    torch::Tensor slice = inputTensorPadded.index(
                        {n, torch::indexing::Slice(), torch::indexing::Slice(startDim1, endDim1),
                         torch::indexing::Slice(startDim2, endDim2)}).to(torch::kCUDA);

                    Conv2d::gradWeight.index_put_({outChannel}, Conv2d::gradWeight.index({outChannel}) +
                                                            slice * errorTensor.index({n, outChannel, i, j}));
                    Conv2d::gradBias.index_put_({outChannel},
                                        gradBias.index({outChannel}) + errorTensor.index({n, outChannel, i, j}));
                    gradInput.index_put_({n, torch::indexing::Slice(), torch::indexing::Slice(startDim1, endDim1),
                                          torch::indexing::Slice(startDim2, endDim2)},
                                         errorTensor.index({n, outChannel, i, j}) * Conv2d::weights.index({outChannel}) +
                                             gradInput.index({n, torch::indexing::Slice(),
                                                              torch::indexing::Slice(startDim1, endDim1),
                                                              torch::indexing::Slice(startDim2, endDim2)}));
                }
            }
        }
        backwardOutput.index_put_({n}, Conv2d::removePad(gradInput.index({n})));
    }

    optimizer->update(Conv2d::weights, Conv2d::gradWeight);
    // Common mistake: pruning the bias usually harms model accuracy too much. (https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide#:~:text=Common%20mistake%3A%20pruning%20the%20bias%20usually%20harms%20model%20accuracy%20too%20much.)

    errorTensor = backwardOutput;
}