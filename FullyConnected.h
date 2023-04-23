//
// Created by hassan on 22.04.23.
//

#ifndef CNN_IN_C___FULLYCONNECTED_H
#define CNN_IN_C___FULLYCONNECTED_H

#endif //CNN_IN_C___FULLYCONNECTED_H

#include "Base.h"
#include "Optimizer.h"

#include "unsupported/Eigen/CXX11/Tensor"
#include "iostream"

class Linear: public  BaseLayer{
public: Linear(int inFeatures, int outFeatures) : inputSize(inFeatures), outputSize(outFeatures){
        trainable = true;
        initializable = true;

        weights(inFeatures, outFeatures);
        weights.setRandom();

//        Eigen::Tensor<float, 1> bias(outFeatures);
//        bias(inFeatures);
//        bias.setConstant(1);
    }

    Eigen::Tensor<float, 2> forward(const Eigen::Tensor<float, 2> & inputTensor) {
        return inputTensor * weights;
}

private:
    int inputSize;
    int outputSize;
    Eigen::Tensor<float, 2> weights;
    Eigen::Tensor<float, 1> bias;
public:
    Optimizer optimizer;
};

