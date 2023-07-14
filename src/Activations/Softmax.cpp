#include "Softmax.hpp"
#include "iostream"

using namespace DeepStorm::Activations;

SoftMax::SoftMax()
{
    SoftMax::trainable = false;
    SoftMax::initializable = false;
}

torch::Tensor SoftMax::forward(torch::Tensor &x) 
{
    auto maxVal = x.amax({-1}, true);
    auto exp_x = torch::exp(x - maxVal);
    auto sum_exp_x = exp_x.sum(-1, true);

    SoftMax::softmaxOutput = exp_x / sum_exp_x;
    return SoftMax::softmaxOutput;
}

torch::Tensor SoftMax::backward(torch::Tensor &y) 
{
    const auto batch_size = SoftMax::softmaxOutput.size(0);
    const auto num_classes = SoftMax::softmaxOutput.size(1);
    auto jacobian = torch::zeros({batch_size, num_classes, num_classes}, torch::kCUDA);
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < num_classes; ++j)
        {
            for (int k = 0; k < num_classes; ++k)
            {
                if (j == k)
                {
                    jacobian[i][j][k] = SoftMax::softmaxOutput[i][j] * (1 - SoftMax::softmaxOutput[i][j]);
                }
                else
                {
                    jacobian[i][j][k] = -SoftMax::softmaxOutput[i][j] * SoftMax::softmaxOutput[i][k];
                }
            }
        }
    }
    auto out = torch::matmul(jacobian, y.unsqueeze(-1)).squeeze();
    return out;
}