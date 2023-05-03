
#include "Base.h"
#include <torch/torch.h>
#include "iostream"

class ReLU : public BaseLayer
{

public:
    ReLU()
    {
        trainable = false;
        initializable = false;
    }

    torch::Tensor forward(torch::Tensor &x)
    {
        pos = x.greater_(0);
        return torch::max(x, torch::zeros_like(x, torch::kCUDA));
    }

    torch::Tensor backward(torch::Tensor &y)
    {
        return pos * y;
    }

private:
    torch::Tensor pos;
};