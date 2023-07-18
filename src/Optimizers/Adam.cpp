#include "Adam.hpp"

using namespace DeepStorm::Optimizers;

Adam::Adam(float learningRate = 0.001, float mu = 0.9, float rho = 0.9, float epsilon = 1e-07)
{
    Adam::learningRate = learningRate;
    Adam::mu = mu;
    Adam::rho = rho;
    Adam::epsilon = epsilon; 
}

torch::Tensor Adam::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor)
{
    Adam::k = Adam::k + 1;

    Adam::v = Adam::mu * Adam::v + (1 - Adam::mu) * gradientTensor;
    Adam::r = Adam::rho * Adam::r + (1 - Adam::rho) * gradientTensor.pow(2);

    auto vHat = (Adam::v) / (1 - pow(Adam::mu, Adam::k));
    auto rHat = (Adam::r) / (1 - pow(Adam::rho, Adam::k));

    return weightTensor - Adam::learningRate * (vHat) / (pow(rHat, 0.5) + Adam::epsilon);
}