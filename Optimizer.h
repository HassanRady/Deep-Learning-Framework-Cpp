//
// Created by hassan on 22.04.23.
//


#include <torch/torch.h>


class Optimizer {
public: Optimizer() {}
public: Optimizer(double learningRate) : learningRate(learningRate){}

protected:
    double learningRate;
};

class Sgd: public Optimizer{

public: Sgd(double learningRate) : Optimizer(learningRate) {}

torch::Tensor update(torch::Tensor & weightTensor, const torch::Tensor & gradientTensor) {
    weightTensor = weightTensor - learningRate * gradientTensor;
    return weightTensor;
}

};