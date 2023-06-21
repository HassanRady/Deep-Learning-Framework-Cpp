#pragma once

#include <iostream>
#include <torch/torch.h>
#include "Base.h"

class CrossEntropyLoss: public BaseLayer {
public:
    CrossEntropyLoss(float epsilon=1e-09) {
        this->epsilon = epsilon;
        trainable = false;
        initializable = false;
    }

    float forward(torch::Tensor & y_hat, torch::Tensor & y)  override{
        this->y = y;
        auto product = y * torch::log(y_hat + epsilon);
        float output = -product.sum().item<float>();
        std::cout << output << "\n";
        return output;
    }

    torch::Tensor backward(torch::Tensor & grad_y) override{
        return torch::where(grad_y.eq(1), -grad_y/(y + epsilon), 0);
    }

private:
    torch::Tensor y;
    float epsilon;
};