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

    torch::Tensor forward(torch::Tensor & y_hat, torch::Tensor & y) {
        this->y = y;
        auto product = y * torch::log(y_hat + epsilon);
        return -torch::sum(product);
    }

    torch::Tensor backward(torch::Tensor & grad_y) {
        return torch::where(grad_y.eq(1), -grad_y/(y + epsilon), 0);
    }

private:
    torch::Tensor y;
    float epsilon;
};