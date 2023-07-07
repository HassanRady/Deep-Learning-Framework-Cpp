#include "Model.h"

using namespace DeepStorm;

Model::Model(std::vector<BaseLayer *> layers) : layers(layers) {}

void Model::append(BaseLayer *layer)
{
    layers.emplace_back(layer);
}

torch::Tensor Model::forward(torch::Tensor x)
{
    for (auto layer : layers)
        x = layer->forward(x);
    return x;
}

void Model::backward(torch::Tensor y)
{
    for (int i = layers.size() - 1; i >= 0; --i)
        y = layers[i]->backward(y);
}

void Model::eval()
{
    for (auto layer : layers)
        layer->eval();
}

void Model::train()
{
    for (auto layer : layers)
        layer->train();
}