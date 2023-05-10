#include <iostream>
#include <torch/torch.h>
#include "Base.h"

class Dropout: public BaseLayer {
public:
    Dropout(float probability=0.5): probability(probability) {
        trainable = false;
        initializable = false;
    }

    torch::Tensor forward(troch::Tensor & inputTensor) {
        if (testingPhase)
            return inputTensor;

//        mask = torch::rand({inputTensor.index()})
        return inputTensor;
    }

    torch::Tensor backward(torch::Tensor & errorTensor) {

        return errorTensor;
    }

private:
    float probability;
    bool testingPhase = false;
    torch::Tensor mask;
};