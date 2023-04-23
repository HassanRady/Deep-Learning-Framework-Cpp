//
// Created by hassan on 22.04.23.
//

#ifndef CNN_IN_C___OPTIMIZER_H
#define CNN_IN_C___OPTIMIZER_H

#endif //CNN_IN_C___OPTIMIZER_H

#include "Eigen/Dense"

class Optimizer {
public: Optimizer() {}
public: Optimizer(double learningRate) : learningRate(learningRate){}

protected:
    double learningRate;
};

class Sgd: public Optimizer{

public: Sgd(double learningRate) : Optimizer(learningRate) {}

Eigen::MatrixXd update(Eigen::MatrixXd & weightTensor, const Eigen::MatrixXd & gradientTensor) {
    weightTensor = weightTensor - learningRate * gradientTensor;
    return weightTensor;
}

};