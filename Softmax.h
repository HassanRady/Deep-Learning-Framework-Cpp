//
// Created by hassan on 03.05.23.
//

#include "Base.h"
#include <torch/torch.h>
#include "iostream"

class SoftMax: public BaseLayer {
public:
    SoftMax() {
        trainable = false;
        initializable = false;
    }

    torch::Tensor forward(torch::Tensor & x) {
        auto max_val = x.amax({-1}, true);
        auto exp_x = torch::exp(x - max_val);
        auto sum_exp_x = exp_x.sum(-1, true);
        softmaxOutput = exp_x / sum_exp_x;
        return softmaxOutput;
    }

    torch::Tensor backward(torch::Tensor & y) {

    }

private:
    torch::Tensor softmaxOutput;
};

