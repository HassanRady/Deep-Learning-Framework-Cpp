#include "torch/torch.h"
#include "iostream"

#include "Base.h"
#include "Optimizer.h"

class BatchNorm2d : public BaseLayer {
public:
    BatchNorm2d(int numFeatures, float eps = 1e-11, float momentum = 0.8) {
        trainable = true;
        initializable = false;

        this->numFeatures = numFeatures;
        this->eps = eps;
        this->momentum = momentum;

        gamma = torch::ones({numFeatures}, torch::kCUDA);
        beta = torch::zeros({numFeatures}, torch::kCUDA);

        mean = torch::zeros({numFeatures}, torch::kCUDA);
        variance = torch::ones({numFeatures}, torch::kCUDA);
    }

    torch::Tensor reshapeTensorForInput(torch::Tensor tensor) {
        return tensor;
    }

    torch::Tensor reshapeTensorForOutput(torch::Tensor tensor) {
        return tensor;
    }

    torch::Tensor normalizeTrain(torch::Tensor &tensor) {
        torch::Tensor batchMean = tensor.mean({0});
        torch::Tensor batchVariance = tensor.var({0});
        torch::Tensor batchStd = torch::sqrt(batchVariance + eps);

        inputTensorNormalized = (tensor - batchMean) / batchStd;
        torch::Tensor inputBatchNormalized = gamma * inputTensorNormalized + beta;

        if (torch::eq(mean, 0).item<float>() == 0) {
            mean = batchMean;
            variance = batchVariance;
        } else {
            mean = momentum * mean + (1 - momentum) * batchMean;
            variance = momentum * variance + (1 - momentum) * batchVariance;
        }

        return inputBatchNormalized;
    }

    torch::Tenor normalizeTest(torch::Tensor &tensor) {
        torch::Tensor inputTensorNormalize = (tensor - mean) / torch::sqrt(variance + eps);
        torch::Tensor inputNormalized = gamma * inputTensorNormalize + beta;
        return inputNormalized;
    }

    torch::Tenosr forward(torch::Tensor &inputTensor) {
        this->inputTensor = inputTensor;
        batchSize = inputTensor.sizes()[0];

        torch::Tensor inputTensorReshaped = reshapeTensorForInput(inputTensor);

        torch::Tensor inputNormalized;

        if(!testPhase)
            inputNormalized = normalizeTrain(inputTensorReshaped);
        else
            inputNormalized = normalizeTest(inputTensorReshaped);

        torch::Tensor forwardOutputReshaped = reshapeTensorForOutput(inputNormalized);

        return forwardOutputReshaped;
    }

private:
    int batchSize;

    int numFeatures;
    float eps;
    float momentum;

    torch::Tensor gamma;
    torch::Tensor beta;

    torch::Tensor mean;
    torch::Tensor variance;

    torch::Tensor inputTensor;
    torch::Tensor inputTensorNormalized;

    Optimizer optimizer;

    bool testPhase;
};