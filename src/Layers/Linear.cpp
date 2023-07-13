#include "Linear.hpp"

using namespace DeepStorm::Layers;

Linear::Linear(int inFeatures, int outFeatures, WeightInitializer *weightInitializer, WeightInitializer *biasInitializer, Optimizer * optimizer)
{
    trainable = true;
    initializable = true;

    Linear::inFeatures = inFeatures;
    Linear::outFeatures = outFeatures;

    Linear::weightInitializer = weightInitializer;
    Linear::biasInitializer = biasInitializer;

    Linear::optimizer = optimizer;

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

void Linear::forward(torch::Tensor &x) 
{
    auto inputDims = x.sizes();
    batchSize = inputDims[0];

    Linear::inputTensor = torch::cat({x, torch::ones({batchSize, 1}, torch::kCUDA)}, -1);
    x = torch::matmul(Linear::inputTensor, Linear::weights);
}

void Linear::backward(torch::Tensor &errorTensor) 
{
    Linear::gradientWeights = torch::matmul(Linear::inputTensor.transpose(1, 0), errorTensor);

    optimizer->update(Linear::weights, Linear::gradientWeights);
    errorTensor = torch::matmul(errorTensor, Linear::weights.transpose(1, 0)).index({torch::indexing::Slice(), torch::indexing::Slice(0, Linear::inFeatures)});
}