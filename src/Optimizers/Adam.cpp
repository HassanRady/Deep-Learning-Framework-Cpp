#include "Adam.hpp"

using namespace DeepStorm::Optimizers;

Adam::Adam(double learningRate = 0.001, double mu = 0.9, double rho = 0.9, double epsilon = 1e-07)
{
    Adam::learningRate = learningRate;
    Adam::mu = mu;
    Adam::rho = rho;
    Adam::epsilon = epsilon; 
}

void Adam::update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor)
{
    k = k + 1;

    v = mu * v + (1 - mu) * gradientTensor;
    r = rho * r + (1 - rho) * gradientTensor.pow(2);

    v = mu * v + (1 - mu) * gradientTensor;
    r = rho * r + (1 - rho) * gradientTensor.pow(2);

    auto vHat = (v) / (1 - pow(mu, k));
    auto rHat = (r) / (1 - pow(rho, k));

    weightTensor = weightTensor - Adam::learningRate * (vHat) / (pow(rHat, 2) + Adam::epsilon);
}