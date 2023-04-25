//
// Created by hassan on 22.04.23.
//

#include <torch/torch.h>

class Optimizer {
public:
    Optimizer() {}

    Optimizer(double learningRate) : learningRate(learningRate) {}

    virtual torch::Tensor update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) = 0;

protected:
    double learningRate;
};

class Sgd : public Optimizer {
public:
    Sgd(double learningRate) : Optimizer(learningRate) {}

    Sgd() {}

    torch::Tensor update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override {
        weightTensor = weightTensor - learningRate * gradientTensor;
        return weightTensor;
    }
};


class SgdWithMomentum : public Optimizer {
public:
    SgdWithMomentum(double learningRate, double momentum) : Optimizer(learningRate), momentum(momentum) {}

    torch::Tensor update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override {
        bool firstFlag = true;
        if (firstFlag) {
            auto v = momentum * vInit - learningRate * gradientTensor;
            firstFlag = false;
        } else {
            v = momentum * v - learningRate * gradientTensor;
        }
        weightTensor = weightTensor + v;
        return weightTensor;
    }

private:
    double momentum;
    double vInit = 0.0;
};

class Adam : public Optimizer {
public:
    Adam(double learningRate = 0.001, double mu = 0.9, double rho = 0.9, double epsilon = 1e-07) : Optimizer(
            learningRate), mu(mu), rho(rho),
                                                                                                   epsilon(epsilon) {}

    torch::Tensor update(torch::Tensor &weightTensor, const torch::Tensor &gradientTensor) override {
        bool firstFlag = true;
        k = k + 1;
        if (firstFlag) {
            auto v = mu * vInit + (1 - mu) * gradientTensor;
            auto r = rho * rInit + (1 - rho) * gradientTensor.pow(2);
            firstFlag = false;
        } else {
            v = mu * v + (1 - mu) * gradientTensor;
            r = rho * r + (1 - rho) * gradientTensor.pow(2);
        }
        auto vHat = (v) / (1 - pow(mu, k));
        auto rHat = (r) / (1 - pow(rho, k));

        weightTensor = weightTensor - learningRate * (vHat) / (pow(rHat, 2) + epsilon);
        return weightTensor;
    }

private:
    double mu, rho, epsilon, k = 0.0, vInit = 0.0, rInit = 0.0;
};