#include "BatchNormalization.h"

using namespace DeepStorm::Layers;

BatchNorm2d::BatchNorm2d(int numFeatures, float eps = 1e-11, float momentum = 0.8)
{
    trainable = true;
    initializable = false;

    this->numFeatures = numFeatures;
    this->eps = eps;
    this->momentum = momentum;

    weight = torch::ones({numFeatures}, torch::kCUDA);
    bias = torch::zeros({numFeatures}, torch::kCUDA);

    mean = torch::zeros({numFeatures}, torch::kCUDA);
    variance = torch::ones({numFeatures}, torch::kCUDA);
}

torch::Tensor BatchNorm2d::normalizeTrain(torch::Tensor &tensor)
{
    torch::Tensor batchMean = tensor.mean({0, 2, 3}, false);
    torch::Tensor batchVariance = tensor.var({0, 2, 3}, false);
    torch::Tensor batchStd = torch::sqrt(batchVariance + eps);
    auto n = tensor.numel() / tensor.sizes()[1];

    mean = momentum * batchMean + (1 - momentum) * mean;
    variance = momentum * batchVariance * n / (n - 1) + (1 - momentum) * variance;

    inputTensorNormalized = (tensor - batchMean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) /
                            batchStd.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    torch::Tensor inputBatchNormalized = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * inputTensorNormalized +
                                         bias.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    return inputBatchNormalized;
}

torch::Tensor BatchNorm2d::normalizeTest(torch::Tensor &tensor)
{
    torch::Tensor inputTensorNormalize = (tensor - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) /
                                         torch::sqrt(variance.unsqueeze(0).unsqueeze(2).unsqueeze(3) + eps);
    torch::Tensor inputNormalized = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * inputTensorNormalize +
                                    bias.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    return inputNormalized;
}

torch::Tensor BatchNorm2d::forward(torch::Tensor &inputTensor) override
{
    this->inputTensor = inputTensor;
    batchSize = inputTensor.sizes()[0];

    torch::Tensor inputNormalized;

    if (training)
        inputNormalized = normalizeTrain(inputTensor);
    else
        inputNormalized = normalizeTest(inputTensor);

    return inputNormalized;
}

torch::Tensor BatchNorm2d::backward(torch::Tensor &errorTensor) override
{
    torch::Tensor gradientWeight = torch::sum(errorTensor * inputTensorNormalized, {0, 2, 3});
    torch::Tensor gradientBias = errorTensor.sum({0, 2, 3});
    torch::Tensor gradientInputNormalized = errorTensor * weight.unsqueeze(0).unsqueeze(2).unsqueeze(3);

    torch::Tensor gradientInputTensor =
        1 / (batchSize * torch::sqrt(variance.unsqueeze(0).unsqueeze(2).unsqueeze(3) + eps) *
                 (batchSize * gradientInputNormalized - gradientInputNormalized.sum({0})) -
             inputTensorNormalized * torch::sum(gradientInputNormalized * inputTensorNormalized, {0}));

    optimizer->update(weight, gradientWeight);
    optimizer->update(bias, gradientBias);

    return gradientInputTensor;
}