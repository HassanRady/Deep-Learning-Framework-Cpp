#include <iostream>
#include <torch/torch.h>
#include "Base.h"

class Flatten: public BaseLayer{
public:
    Flatten() {
        trainable = false;
        initializable = false;
    }

    torch::Tensor forward(torch::Tensor & inputTensor) override{
        this->inputTensor = inputTensor;
        auto inputTensorDims = inputTensor.sizes();
        int denseFeatures = 1;
        for(int dim=1; dim < inputTensorDims.size();++dim) {
            denseFeatures *= inputTensorDims[dim];
        }
        auto flattenTensor = inputTensor.reshape({inputTensor.sizes()[0], inputTensor.sizes()[1] * inputTensor.sizes()[2] * inputTensor.sizes()[3]});
        std::cout << flattenTensor.sizes() << "\n";
        return flattenTensor;
    }

    torch::Tensor backward(torch::Tensor & errorTensor) override{
        auto reshapedTensor = errorTensor.reshape(inputTensor.sizes());
        return reshapedTensor;
    }

private:
    torch::Tensor inputTensor;
};