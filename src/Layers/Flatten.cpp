#include "Flatten.hpp"

using namespace DeepStorm::Layers;

Flatten::Flatten()
{
    trainable = false;
    initializable = false;
}

torch::Tensor Flatten::forward(torch::Tensor &inputTensor) 
{
    this->inputTensor = inputTensor;
    auto inputTensorDims = inputTensor.sizes();
    int denseFeatures = 1;
    for (int dim = 1; dim < inputTensorDims.size(); ++dim)
    {
        denseFeatures *= inputTensorDims[dim];
    }
    auto flattenTensor::flattenTensor = inputTensor.reshape({inputTensor.sizes()[0], inputTensor.sizes()[1] * inputTensor.sizes()[2] * inputTensor.sizes()[3]});
    return flattenTensor;
}

torch::Tensor Flatten::backward(torch::Tensor &errorTensor) 
{
    auto reshapedTensor = errorTensor.reshape(inputTensor.sizes());
    return reshapedTensor;
}