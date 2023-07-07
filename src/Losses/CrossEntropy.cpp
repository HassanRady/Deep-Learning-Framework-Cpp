#include "CrossEntropy.hpp"

using namespace DeepStorm::Losses;

CrossEntropyLoss::CrossEntropyLoss(float epsilon = 1e-09)
{
    this->epsilon = epsilon;
}

float CrossEntropyLoss::forward(torch::Tensor &y_hat, torch::Tensor &y) 
{
    this->y = y;
    auto product = y * torch::log(y_hat + epsilon);
    return -product.sum().item<float>();
}

torch::Tensor CrossEntropyLoss::backward(torch::Tensor &grad_y) 
{
    return torch::where(grad_y.eq(1), -grad_y / (y + epsilon), 0);
}