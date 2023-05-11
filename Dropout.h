#include <iostream>
#include <torch/torch.h>
#include "Base.h"

class Dropout : public BaseLayer {
public:
    Dropout(float probability = 0.5) : probability(probability) {
        trainable = false;
        initializable = false;
    }

    torch::Tensor forward(torch::Tensor &inputTensor) {
        if (testingPhase)
            return inputTensor;

        auto tensorShape = inputTensor.sizes();
        mask = torch::rand({tensorShape[tensorShape.size() - 2], tensorShape[tensorShape.size() - 1]}).to(inputTensor.device());
        return mask * inputTensor;
    }

    torch::Tensor backward(torch::Tensor &errorTensor) {
        auto out = errorTensor * mask;
        out = out / probability;
        return out;
    }

private:
    float probability;
    bool testingPhase = false;
    torch::Tensor mask;
};