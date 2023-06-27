#pragma once

#include <iostream>
#include <torch/torch.h>

class Loss {
    public:
    Loss(){}
    virtual float forward(torch::Tensor &y_hat, torch::Tensor &y) = 0;
    virtual torch::Tensor backward(torch::Tensor &grad_y)= 0;

};

class CrossEntropyLoss: public Loss
{
public:
    CrossEntropyLoss(float epsilon = 1e-09)
    {
        this->epsilon = epsilon;
    }

    float forward(torch::Tensor &y_hat, torch::Tensor &y) override
    {
        this->y = y;
        auto product = y * torch::log(y_hat + epsilon);
        return -product.sum().item<float>();
    }

    torch::Tensor backward(torch::Tensor &grad_y) override
    {
        return torch::where(grad_y.eq(1), -grad_y / (y + epsilon), 0);
    }

private:
    torch::Tensor y;
    float epsilon;
};