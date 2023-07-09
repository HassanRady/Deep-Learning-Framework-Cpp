#include "Model.hpp"
#include "iostream"

using namespace DeepStorm;

Model::Model() {}

Model::Model(std::vector<Layer *> layers)
{
    Model::layers = layers;
}

void Model::append(Layer *layer)
{
    Model::layers.emplace_back(layer);
}

torch::Tensor Model::forward(torch::Tensor x)
{
    for (auto it = Model::layers.begin(); it != layers.end(); it++)
    {
        x = (*it)->forward(x);
    }
    return x;
}

void Model::backward(torch::Tensor y)
{
    for (auto it = Model::layers.rbegin(); it != Model::layers.rend(); it++)
    {
        y = (*it)->backward(y);
    }
}

void Model::eval()
{
    for (auto it = Model::layers.begin(); it != layers.end(); it++)
        (*it)->eval();
}

void Model::train()
{
    for (auto it = Model::layers.begin(); it != layers.end(); it++)
        (*it)->train();
}