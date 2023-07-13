#include "BatchNormalization.hpp"
#include "iostream"

using namespace DeepStorm::Layers;

BatchNorm2d::BatchNorm2d(int numFeatures, Optimizer * optimizer, float eps = 1e-11, float momentum = 0.8)
{
    trainable = true;
    initializable = false;

    BatchNorm2d::numFeatures = numFeatures;
    BatchNorm2d::eps = eps;
    BatchNorm2d::momentum = momentum;

    BatchNorm2d::weight = torch::ones({numFeatures}, torch::kCUDA);
    BatchNorm2d::bias = torch::zeros({numFeatures}, torch::kCUDA);

    BatchNorm2d::mean = torch::zeros({numFeatures}, torch::kCUDA);
    BatchNorm2d::variance = torch::ones({numFeatures}, torch::kCUDA);

    BatchNorm2d::optimizer = optimizer;
}

torch::Tensor BatchNorm2d::normalizeTrain(torch::Tensor &x)
{
    torch::Tensor batchMean = x.mean({0, 2, 3}, false);
    torch::Tensor batchVariance = x.var({0, 2, 3}, false);
    torch::Tensor batchStd = torch::sqrt(batchVariance + BatchNorm2d::eps);
    auto n = x.numel() / x.sizes()[1];

    mean = BatchNorm2d::momentum * batchMean + (1 - BatchNorm2d::momentum) * BatchNorm2d::mean;
    variance = BatchNorm2d::momentum * batchVariance * n / (n - 1) + (1 - BatchNorm2d::momentum) * BatchNorm2d::variance;

    BatchNorm2d::inputTensorNormalized = (x - batchMean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) /
                            batchStd.unsqueeze(0).unsqueeze(2).unsqueeze(3);

    x = BatchNorm2d::weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * BatchNorm2d::inputTensorNormalized +
                                         BatchNorm2d::bias.unsqueeze(0).unsqueeze(2).unsqueeze(3);
}

torch::Tensor BatchNorm2d::normalizeTest(torch::Tensor &tensor)
{
    torch::Tensor inputTensorNormalize = (tensor - BatchNorm2d::mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) /
                                         torch::sqrt(variance.unsqueeze(0).unsqueeze(2).unsqueeze(3) + eps);
    torch::Tensor inputNormalized = BatchNorm2d::weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * inputTensorNormalize +
                                    BatchNorm2d::bias.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    return inputNormalized;
}

void BatchNorm2d::forward(torch::Tensor &x) 
{
    BatchNorm2d::inputTensor = x;
    BatchNorm2d::batchSize = x.sizes()[0];

    torch::Tensor inputNormalized;

    if (BatchNorm2d::training)
        BatchNorm2d::normalizeTrain(x);
    else
        BatchNorm2d::normalizeTest(x);

}

void BatchNorm2d::backward(torch::Tensor &errorTensor) 
{
    torch::Tensor gradientWeight = torch::sum(errorTensor * BatchNorm2d::inputTensorNormalized, {0, 2, 3});
    torch::Tensor gradientBias = errorTensor.sum({0, 2, 3});
    torch::Tensor gradientInputNormalized = errorTensor * BatchNorm2d::weight.unsqueeze(0).unsqueeze(2).unsqueeze(3);

    errorTensor =
        1 / (BatchNorm2d::batchSize * torch::sqrt(BatchNorm2d::variance.unsqueeze(0).unsqueeze(2).unsqueeze(3) + BatchNorm2d::eps)) *
                 (BatchNorm2d::batchSize * gradientInputNormalized - gradientInputNormalized.sum({0}) -
             BatchNorm2d::inputTensorNormalized * torch::sum(gradientInputNormalized * BatchNorm2d::inputTensorNormalized, {0}));

    BatchNorm2d::optimizer->update(BatchNorm2d::weight, gradientWeight);
    BatchNorm2d::optimizer->update(BatchNorm2d::bias, gradientBias);
}