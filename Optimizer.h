#pragma once

#include "iostream"
#include <torch/torch.h>

class Optimizer
{
public:
    Optimizer() {}

    Optimizer(double learningRate) : learningRate(learningRate) {}

    virtual int update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) = 0;
    virtual ~Optimizer() = 0;

public:
    double learningRate;
};

Optimizer::~Optimizer() {
    std::cout << "Pure virtual destructor is called \n";
}

class Sgd : public Optimizer
{
public:
    Sgd(double learningRate) : Optimizer(learningRate) {}

    Sgd() {}

    int update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override
    {
        weightTensor = weightTensor - learningRate * gradientTensor;
        return 0;
    }

    ~Sgd()
    {
        std::cout << "sgd deleted"
                  << "\n";
    }
};

class SgdWithMomentum : public Optimizer
{
public:
    SgdWithMomentum(double learningRate, double momentum) : Optimizer(learningRate), momentum(momentum) {}

    int update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override
    {
        v = momentum * v - learningRate * gradientTensor;
        weightTensor = weightTensor + v;
        return 0;
    }
    ~SgdWithMomentum()
    {
        std::cout << "SgdWithMomentum deleted"
                  << "\n";
    }

private:
    double momentum;
    torch::Tensor v = torch::zeros({1}, torch::kCUDA);
};

class Adam : public Optimizer
{
public:
    Adam(double learningRate = 0.001, double mu = 0.9, double rho = 0.9, double epsilon = 1e-07) : Optimizer(
                                                                                                       learningRate),
                                                                                                   mu(mu), rho(rho),
                                                                                                   epsilon(epsilon) {}

    int update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override
    {
        k = k + 1;

        v = mu * v + (1 - mu) * gradientTensor;
        r = rho * r + (1 - rho) * gradientTensor.pow(2);

        v = mu * v + (1 - mu) * gradientTensor;
        r = rho * r + (1 - rho) * gradientTensor.pow(2);

        auto vHat = (v) / (1 - pow(mu, k));
        auto rHat = (r) / (1 - pow(rho, k));

        weightTensor = weightTensor - learningRate * (vHat) / (pow(rHat, 2) + epsilon);
        return 0;
    }

    ~Adam()
    {
        std::cout << "Adam deleted"
                  << "\n";
    }

private:
    double mu, rho, epsilon, k = 0.0;
    torch::Tensor v = torch::zeros({1}, torch::kCUDA);
    torch::Tensor r = torch::zeros({1}, torch::kCUDA);
};