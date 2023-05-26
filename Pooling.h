#include "torch/torch.h"
#include "iostream"

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

};