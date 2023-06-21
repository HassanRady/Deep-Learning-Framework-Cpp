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

    virtual torch::Tensor forward(torch::Tensor & tensor) {}
    virtual float forward(torch::Tensor & tensor, torch::Tensor & label) {}

    virtual torch::Tensor backward(torch::Tensor & tensor) {}

    bool trainable;
    bool initializable;
    bool training;
    Optimizer optimizer;
};