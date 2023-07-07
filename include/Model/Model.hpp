#pragma once

#include "torch/torch.h"
#include "iostream"
#include "vector"
#include "string"
#include "tuple"

#include "Layer.hpp"
#include "Optimizer.hpp"

namespace DeepStorm
{
    class Model
    {
    public:
        Model();

        Model(std::vector<Layer *> layers);

        void append(Layer *layer);

        torch::Tensor forward(torch::Tensor x);

        void backward(torch::Tensor y);

        void eval();

        void train();

    private:
        std::vector<Layer *> layers;
        std::vector<Layer *>::iterator iter;
    };
} // namespace DeepStorm
