#include <iostream>
#include <torch/torch.h>
#include "string"

#include "Base.h"
#include "Initializers.h"
#include "Optimizer.h"

class Conv2d: public BaseLayer{
public:
    Conv2d(int inChannels, int outChannels, int kernelSize, int stride, std::string padding, WeightInitializer weightInitializer=He(), WeightInitializer biasInitializer=Constant()) {
        trainable = true;
        initializable = true;

        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->weightInitializer = weightInitializer;
        this->biasInitializer = biasInitializer;

    }

private:
    Optimizer optimizer;
    int inChannels;
    int outChannels;
    int kernelSize;
    int stride;
    std::string padding;
    WeightInitializer weightInitializer;
    WeightInitializer biasInitializer;

};