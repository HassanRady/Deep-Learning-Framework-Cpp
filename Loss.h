#include <iostream>
#include <torch/torch.h>
#include "Base.h"

class CrossEntropyLoss: public BaseLayer {
public:
    CrossEntropyLoss(float epsilon=1e-07) {
        this->epsilon = epsilon;
        trainable = false;
        initializable = false;
    }

    torch::Tensor forward(torch::Tensor & y_hat, torch::Tensor & y) {
        this->y = y;
        auto product = y * torch::log(y_hat + epsilon);
        return -torch::sum(product);
    }

private:
    torch::Tensor y;
    float epsilon;
};