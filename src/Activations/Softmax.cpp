#include "Softmax.hpp"
#include "iostream"

using namespace DeepStorm::Activations;

SoftMax::SoftMax()
{
    SoftMax::trainable = false;
    SoftMax::initializable = false;
    SoftMax::training = false;
    SoftMax::name = "SoftMax";
}

torch::Tensor SoftMax::forward(torch::Tensor &x) 
{
    auto maxVal = x.amax({-1}, true);
    auto exp_x = torch::exp(x - maxVal);
    auto sum_exp_x = exp_x.sum(-1, true);
    SoftMax::softmaxOutput = exp_x / sum_exp_x;
    return SoftMax::softmaxOutput;
}

torch::Tensor SoftMax::backward(torch::Tensor &y) {
    auto softExpand = SoftMax::softmaxOutput.unsqueeze(-1).expand({y.sizes()[0], y.sizes()[1], y.sizes()[1]});
    auto jacobian = softExpand * (torch::eye(y.sizes()[1], torch::kCUDA) - softExpand.transpose(1, 2));
    return torch::matmul(jacobian, y.unsqueeze(-1)).squeeze();
}