#include "torch/torch.h"
#include "iostream"
#include "vector"
#include "string"
#include "tuple"

#include "Base.h"
#include "Optimizer.h"
#include "Loss.h"

class Model {
public:
    Model(){}

    std::tuple<torch::Tensor, torch::Tensor> trainStep(torch::Tensor x, torch::Tensor y) {
        torch::Tensor output = forward(x);
        torch::Tensor loss = this->loss.forward(output, y);
        backward(y);
        return {loss, output};
    }

    std::tuple<torch::Tensor, torch::Tensor> valStep(torch::Tensor x, torch::Tensor y) {
        torch::Tensor output = forward(x);
        torch::Tensor loss = this->loss.forward(output, y);
        return {loss, output};
    }

    std::tuple<float, torch::Tensor> trainEpoch() {
        for(auto layer:model)
            layer.train();

        std::vector<torch::Tensor> runningPreds;
        float runningLoss = 0.0;


    }

    torch::Tensor forward(torch::Tensor x) {
        for(auto layer:model)
            x = layer.forward(x);
        return x
    }

    int backward(torch::Tensor y) {
        y = loss.backward(y);
        for(auto layer: std::reverse(model.begin(), model.end()))
            y = layer.backward(y);
    }

    int compile(Optimizer optimizer, BaseLayer loss, int batchSize, std::vector<std::string> metrics) {
        this->batchSize = batchSize;
        this->loss = loss;
        this->metrics = metrics;
        setOptimizer(optimizer);
    }

    int setOptimizer(Optimizer optimizer) {
        for(auto layer:model) {
            if(layer.trainable)
                // TODO check if its a deep copy
                layer.optimizer = optimizer;
        }
    }

private:
    int batchSize;

    BaseLayer loss;

    std::vector<std::string> metrics;

    std::vector<BaseLayer> model;
};