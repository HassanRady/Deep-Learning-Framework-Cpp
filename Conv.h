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
        kernelDim1 = kernelSize;
        kernelDim2 = kernelSize;

        this->stride = stride;
        this->padding = padding;
        this->weightInitializer = weightInitializer;
        this->biasInitializer = biasInitializer;

        initialize();

    }

    void initialize() {
        weights = torch::empty({outChannels, inChannels, kernelDim1, kernelDim2}, torch::kCUDA);
        bias = torch::empty({outChannels, 1}, torch::kCUDA);

        weightInitializer.initialize(weights, inChannels * kernelDim1 * kernelDim2, kernelDim1 * kernelDim2 * outChannels);
        biasInitializer.initialize(bias, outChannels, 1);
    }

    int getShapeAfterConv(int dimSize, int kernelSize, int pad, int stride) {
        int startPad = pad;
        int endPad = pad;
        return (int) 1 + (dimSize - kernelSize + startPad + endPad) / stride;
    }

    torch::Tensor convolve(torch::Tensor & slice, torch::Tensor kernel, torch::Tensor & bias) {
        return torch::sum(slice * kernel) + bias;
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
    torch::Tensor bias;

};