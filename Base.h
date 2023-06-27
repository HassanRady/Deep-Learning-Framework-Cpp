#pragma once

#include "torch/torch.h"
#include "Optimizer.h"

class BaseLayer {
public:
    BaseLayer() {
        trainable = false;
        initializable = false;
        training = true;
    }

    void train() {
        training = true;
    }

    void eval() {
        training = false;
    }

    virtual torch::Tensor forward(torch::Tensor & tensor) = 0;
    // virtual float forward(torch::Tensor & tensor, torch::Tensor & label) = 0;

    virtual torch::Tensor backward(torch::Tensor & tensor) = 0;

    bool trainable;
    bool initializable;
    bool training;
    Optimizer* optimizer;
};