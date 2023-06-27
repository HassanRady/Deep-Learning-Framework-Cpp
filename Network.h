#pragma once

#include "torch/torch.h"
#include "iostream"
#include "vector"
#include "string"
#include "tuple"

#include "Base.h"
#include "Optimizer.h"

class Network {
public:
    Network(std::vector<BaseLayer*> layers): layers(layers) {}

    void append(BaseLayer* layer) {
        layers.push_back(layer);
    }

    torch::Tensor forward(torch::Tensor x) {
        for (auto layer: layers)
            x = layer->forward(x);
        return x;
    }

    void backward(torch::Tensor y) {
        for (int i = layers.size() - 1; i >= 0; --i)
            y = layers[i]->backward(y);
    }

    void setOptimizer(Optimizer* optimizer) {
        for (auto layer: layers) {
            if (layer->trainable)
                // TODO check if its a deep copy
                layer->optimizer = optimizer;
        }
    }

    void eval() {
        for (auto layer: layers)
            layer->eval();
    }

    void train() {
        for (auto layer: layers)
            layer->train();
    }

private:
    std::vector <BaseLayer*> layers;
};