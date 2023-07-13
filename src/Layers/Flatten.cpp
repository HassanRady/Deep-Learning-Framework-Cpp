#include "Flatten.hpp"

using namespace DeepStorm::Layers;

Flatten::Flatten()
{
    trainable = false;
    initializable = false;
}

void Flatten::forward(torch::Tensor &x) 
{
    Flatten::inputTensor = x;
    auto inputTensorDims = x.sizes();
    int denseFeatures = 1;
    for (int dim = 1; dim < inputTensorDims.size(); ++dim)
    {
        denseFeatures *= inputTensorDims[dim];
    }
    x = x.reshape({x.sizes()[0], x.sizes()[1] * x.sizes()[2] * x.sizes()[3]});
}

void Flatten::backward(torch::Tensor &errorTensor) 
{
    errorTensor = errorTensor.reshape(Flatten::inputTensor.sizes());
}