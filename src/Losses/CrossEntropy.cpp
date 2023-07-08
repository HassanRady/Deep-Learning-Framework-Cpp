#include "CrossEntropy.hpp"

using namespace DeepStorm::Losses;

CrossEntropyLoss::CrossEntropyLoss(float epsilon = 1e-09)
{
    CrossEntropyLoss::epsilon = epsilon;
}

float CrossEntropyLoss::forward(torch::Tensor &y_hat, torch::Tensor &y) 
{
    CrossEntropyLoss::y = y;
    auto product = y * torch::log(y_hat + CrossEntropyLoss::epsilon);
    return -product.sum().item<float>();
}

torch::Tensor CrossEntropyLoss::backward(torch::Tensor &grad_y) 
{
    return torch::where(grad_y.eq(1), -grad_y / (CrossEntropyLoss::y + CrossEntropyLoss::epsilon), 0);
}