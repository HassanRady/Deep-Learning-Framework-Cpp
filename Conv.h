#include <iostream>
#include <torch/torch.h>
#include "string"
#include "vector"

#include "Base.h"
#include "Initializers.h"
#include "Optimizer.h"

class Conv2d : public BaseLayer {
public:
    Conv2d(int inChannels, int outChannels, int kernelSize, int stride, std::string padding,
           WeightInitializer weightInitializer = He(), WeightInitializer biasInitializer = Constant()) {
        trainable = true;
        initializable = true;

        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->weightInitializer = weightInitializer;
        this->biasInitializer = biasInitializer;

        this->initialize();

    }

    void initialize() {
        weights = torch::empty({inFeatures, outFeatures}, torch::kCUDA);
        bias = torch::empty({1, outFeatures}, torch::kCUDA);

        weightInitializer.initialize(weights, inFeatures, outFeatures);
        biasInitializer.initialize(bias, 1, outFeatures);

        weights = torch::cat({weights, bias}, 0);
    }

    int getShapeAfterConv(int dimSize, int kernelSize, int pad, int stride) {
        int startPad = pad;
        int endPad = pad;
        return (int) 1 + (dimSize - kernelSize + startPad + endPad) / stride
    }


private:
    Optimizer optimizer;
    int inChannels;
    int outChannels;

    int kernelSize;
    int kernelDim1;
    int kernelDim2;

    int stride;
    int strideDim1;
    int strideDim2;

    std::string padding;
    WeightInitializer weightInitializer;
    WeightInitializer biasInitializer;

    torch::Tensor weights;
    torch::Tenosr bias;

};