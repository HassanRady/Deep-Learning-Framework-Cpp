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
        const auto batch_size = softmaxOutput.size(0);
        const auto num_classes = softmaxOutput.size(1);
        auto jacobian = torch::zeros({batch_size, num_classes, num_classes}, torch::kCUDA);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                for (int k = 0; k < num_classes; ++k) {
                    if (j == k) {
                        jacobian[i][j][k] = softmaxOutput[i][j] * (1 - softmaxOutput[i][j]);
                    } else {
                        jacobian[i][j][k] = -softmaxOutput[i][j] * softmaxOutput[i][k];
                    }
                }
            }
        }

        auto out = torch::matmul(jacobian, y.unsqueeze(-1));
        return out.squeeze();
    }

private:
    torch::Tensor softmaxOutput;
};

