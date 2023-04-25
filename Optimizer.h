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

    torch::Tensor update(torch::Tensor &weightTensor, torch::Tensor &gradientTensor) override {
        auto v = momentum * v - learningRate * gradientTensor;
        return weightTensor = weightTensor + v;
    }

private:
    double momentum;
}