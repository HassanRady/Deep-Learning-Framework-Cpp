#include "Linear.hpp"

using namespace DeepStorm::Layers;

Linear::Linear(int inFeatures, int outFeatures, WeightInitializer *weightInitializer, WeightInitializer *biasInitializer)
    : inFeatures(inFeatures), outFeatures(outFeatures)
{
    trainable = true;
    initializable = true;

    Linear::weightInitializer = weightInitializer;
    Linear::biasInitializer = biasInitializer;

    Linear::initialize();
}

void Linear::initialize()
{
    Linear::weights = torch::empty({inFeatures, outFeatures}, torch::kCUDA);
    Linear::bias = torch::empty({1, outFeatures}, torch::kCUDA);

    Linear::weightInitializer->initialize(Linear::weights, inFeatures, outFeatures);
    Linear::biasInitializer->initialize(Linear::bias, 1, outFeatures);

    Linear::weights = torch::cat({Linear::weights, Linear::bias}, 0);
}

torch::Tensor Linear::forward(torch::Tensor &inputTensor) 
{
    auto inputDims = inputTensor.sizes();
    batchSize = inputDims[0];

    Linear::inputTensor = torch::cat({inputTensor, torch::ones({batchSize, 1}, torch::kCUDA)}, -1);
    return torch::matmul(Linear::inputTensor, weights);
}

torch::Tensor Linear::backward(torch::Tensor &errorTensor) 
{
    gradientWeights = torch::matmul(inputTensor.transpose(1, 0), errorTensor);

    optimizer->update(Linear::weights, gradientWeights);

    auto out = torch::matmul(errorTensor, Linear::weights.transpose(1, 0)).index({torch::indexing::Slice(), torch::indexing::Slice(0, inFeatures)});

    return out;
}