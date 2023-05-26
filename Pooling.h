#include "torch/torch.h"
#include "iostream"
#include "vector"

#include "Base.h"

class MaxPool2d: public BaseLayer {
public:
    MaxPool2d(torch::ExpandingArray<2> kernelSize, torch::ExpandingArray<2> stride) {
        trainable = false;
        initializable = false;

        kernelSizeDim1 = kernelSize->operator[](0);
        kernelSizeDim2 = kernelSize->operator[](1);

        strideDim1 = stride->operator[](0);
        strideDim2 = stride->operator[](1);
    }

    int getShapeAfterPooling(int dimSize, int kernelSize, int stride) {
        return (int) 1 + (dimSize - kernelSize)/stride;
    }

    std::vector<int> getForwardOutputShape(int inputSizeDim1, int inputSizeDim2) {
        int outputSizeDim1 = getShapeAfterPooling(inputSizeDim1, kernelSizeDim1, strideSizeDim1);
        int outputSizeDim2 = getShapeAfterPooling(inputSizeDim2, kernelSizeDim2, strideSizeDim2);
        return {batchSize, outChannels, outputSizeDim1, outputSizeDim2};
    }

private:
    int batchSize;

    int outChannels;

    int kernelSizeDim1;
    int kernelSizeDim2;

    int strideSizeDim1;
    int strideSizeDim2;
};