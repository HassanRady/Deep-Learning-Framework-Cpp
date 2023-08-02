#include "Linear.hpp"

using namespace DeepStorm::Layers;

Linear::Linear(int inFeatures, int outFeatures, WeightInitializer *weightInitializer, WeightInitializer *biasInitializer, Optimizer * optimizer)
{
    name = "Linear";
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
    Linear::weights = torch::empty({Linear::inFeatures, Linear::outFeatures}, torch::kCUDA);
    Linear::bias = torch::empty({1, Linear::outFeatures}, torch::kCUDA);

    Linear::weightInitializer->initialize(Linear::weights, Linear::inFeatures, Linear::outFeatures);
    Linear::biasInitializer->initialize(Linear::bias, 1, Linear::outFeatures);

    Linear::weights = torch::cat({Linear::weights, Linear::bias}, 0);
}

torch::Tensor Linear::forward(torch::Tensor &inputTensor) 
{
    auto inputDims = inputTensor.sizes();
    Linear::batchSize = inputDims[0];

    Linear::inputTensor = torch::cat({inputTensor, torch::ones({batchSize, 1}, torch::kCUDA)}, -1);

    return torch::matmul(Linear::inputTensor, Linear::weights); 
}

torch::Tensor Linear::backward(torch::Tensor &errorTensor) 
{
    Linear::gradientWeights = torch::matmul(Linear::inputTensor.transpose(1, 0), errorTensor);
    Linear::weights = optimizer->update(Linear::weights, Linear::gradientWeights);
    return torch::matmul(errorTensor, Linear::weights.transpose(1, 0)).index({torch::indexing::Slice(), torch::indexing::Slice(0, Linear::inFeatures)});
}