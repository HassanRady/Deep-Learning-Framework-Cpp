#include "torch/torch.h"
#include "iostream"
#include "vector"
#include "string"
#include "memory"

#include "Layer.hpp"
#include "Conv.hpp"
#include "BatchNormalization.hpp"
#include "Dropout.hpp"
#include "Flatten.hpp"
#include "Linear.hpp"
#include "Pooling.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "He.hpp"
#include "Constant.hpp"
#include "Optimizer.hpp"
#include "Adam.hpp"
#include "ImgDataset.hpp"
#include "Trainer.hpp"
#include "CrossEntropy.hpp"


using namespace std;

using namespace DeepStorm;
using namespace DeepStorm::Layers;
using namespace DeepStorm::Activations;
using namespace DeepStorm::Optimizers;
// using namespace DeepStorm::Initializers;
using namespace DeepStorm::Datasets;
using namespace DeepStorm::Losses;

int main()
{
    torch::manual_seed(1);

    auto batchSize = 2;
    auto inChannels = 1;
    auto filterSize = 3;
    auto outChannels = 16;
    auto stride = 1;
    auto padding = "same";
    auto classes = 10;

    // DeepStorm::WeightInitializer* wInit = &DeepStorm::Initializers::He();

    Initializers::He wInit = Initializers::He();

    DeepStorm::Initializers::Constant bInit(1);

    Conv2d conv1(inChannels, outChannels, filterSize, stride, padding, &wInit, &bInit);
    Conv2d conv2(outChannels, outChannels, filterSize, stride, padding, &wInit, &bInit);
    BatchNorm2d batchNorm1(16, 1e-11, 0.8);
    BatchNorm2d batchNorm2(16, 1e-11, 0.8);
    Dropout dropout(0.3);
    MaxPool2d maxPool1(2, 2);
    MaxPool2d maxPool2(2, 2);
    ReLU relu1;
    ReLU relu2;
    ReLU relu3;
    Flatten flatten;
    Linear fc1(outChannels * 7 * 7, 32, &wInit, &bInit);
    Linear fc2(32, classes, &wInit, &bInit);
    SoftMax softmax;

    Adam optimizer();
    CrossEntropyLoss loss(1e-09);

    auto model = Model(vector<Layer *>{
        &conv1,
        &batchNorm1,
        &dropout,
        &relu1,
        &maxPool1,
        &conv2,
        &batchNorm2,
        &relu2,
        &maxPool2,
        &flatten,
        &fc1,
        &relu3,
        &fc2,
        &softmax});

    conv1.optimizer = new Adam(0.001, 0.9, 0.9, 1e-07);
    conv2.optimizer = new Adam(0.001, 0.9, 0.9, 1e-07);
    batchNorm1.optimizer = new Adam(0.001, 0.9, 0.9, 1e-07);
    batchNorm2.optimizer = new Adam(0.001, 0.9, 0.9, 1e-07);
    fc1.optimizer = new Adam(0.001, 0.9, 0.9, 1e-07);
    fc2.optimizer = new Adam(0.001, 0.9, 0.9, 1e-07);

    auto trainset = ImgDataset("./data/trainset", 1, (unsigned)1);
    auto valset = ImgDataset("./data/trainset", 1, (unsigned)2);

    trainset.resize(12);
    valset.resize(2);

    auto trainLoader = torch::data::make_data_loader(std::move(trainset.map(torch::data::transforms::Stack<>())),
                                                     torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));

    auto valLoader = torch::data::make_data_loader(std::move(valset.map(torch::data::transforms::Stack<>())),
                                                   torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));


    auto trainer = Trainer(model, &loss, batchSize);

    auto [x, y] = trainer.fit<>(*trainLoader, *valLoader, 5);
}