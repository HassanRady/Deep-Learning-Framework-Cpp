#include "Linear.hpp"

using namespace DeepStorm::Layers;

Linear::Linear(int inFeatures, int outFeatures, WeightInitializer *weightInitializer, WeightInitializer *biasInitializer)
    : inFeatures(inFeatures), outFeatures(outFeatures)
{
    trainable = true;
    initializable = true;

    this->weightInitializer = weightInitializer;
    this->biasInitializer = biasInitializer;

    initialize();
}

void Linear::initialize()
{
    weights = torch::empty({inFeatures, outFeatures}, torch::kCUDA);
    bias = torch::empty({1, outFeatures}, torch::kCUDA);

    weightInitializer->initialize(weights, inFeatures, outFeatures);
    biasInitializer->initialize(bias, 1, outFeatures);

    weights = torch::cat({weights, bias}, 0);
}

torch::Tensor Linear::forward(torch::Tensor &inputTensor) override
{
    auto inputDims = inputTensor.sizes();
    batchSize = inputDims[0];

    this->inputTensor = torch::cat({inputTensor, torch::ones({batchSize, 1}, torch::kCUDA)}, -1);
    return torch::matmul(this->inputTensor, weights);
}

torch::Tensor Linear::backward(torch::Tensor &errorTensor) override
{
    gradientWeights = torch::matmul(inputTensor.transpose(1, 0), errorTensor);

    optimizer->update(weights, gradientWeights);

    auto out = torch::matmul(errorTensor, weights.transpose(1, 0)).index({Slice(), Slice(0, inFeatures)});

    return out;
}