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

        // TMP
        this->kernelSize = kernelSize;
        kernelDim1 = kernelSize;
        kernelDim2 = kernelSize;

        // TMP
        this->stride = stride;
        strideDim1 = stride;
        strideDim2 = stride;

        checkPaddingType(padding);

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

    torch::ArrayRef<int> getPadSizeSame(int kernelSize) {
        int startPad;
        if (kernelSize % 2 == 1) {
            startPad = (int) (kernelSize - 1) / 2;
            return {startPad, startPad};
        }
        startPad = (int) kernelSize / 2 - 1;
        int endPad = (int) kernelSize/2;
        return {startPad, endPad};
    }

    void checkPaddingType(std::string padding) {
        if (padding == "same") {
            padSizeDim1 = getPadSizeSame(kernelDim1);
            padSizeDim2 = getPadSizeSame(kernelDim2);
        } else if(padding == "valid") {
            padSizeDim1 = {0, 0};
            padSizeDim2 = {0, 0};
        } else if(isdigit(padding)) {
            padSizeDim1 = {padding, padding};
            padSizeDim2 = {padding, padding};
        }
    }

    torch::Tensor convolve(torch::Tensor & slice, torch::Tensor kernel, torch::Tensor & bias) {
        return torch::sum(slice * kernel) + bias;
    }

    torch::ArrayRef<int> getOutputShape(int inputSizeDim1, int inputSizeDim2) {
        int outputSizeDim1 = getShapeAfterConv(inputSizeDim1, kernelDim1, padSizeDim1, strideDim1);
        int outputSizeDim2 = getShapeAfterConv(inputSizeDim2, kernelDim2, padSizeDim2, strideDim2);
        return {batchSize, outChannels, outputSizeDim1, outputSizeDim2};
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
    torch::ArrayRef<int> padSizeDim1;
    torch::ArrayRef<int> padSizeDim2;

    WeightInitializer weightInitializer;
    WeightInitializer biasInitializer;

    torch::Tensor weights;
    torch::Tensor bias;

    int batchSize;
};