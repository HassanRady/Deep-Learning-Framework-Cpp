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

    Adam optimizer();
    CrossEntropyLoss loss(1e-09);

    Model model = Model();

    model.append(new Conv2d(inChannels, outChannels, filterSize, stride, padding, &wInit, &bInit));
    model.append(new BatchNorm2d(16, 1e-11, 0.8));
    model.append(new Dropout(0.3));
    model.append(new ReLU());
    model.append(new MaxPool2d(2, 2));
    model.append(new Conv2d(outChannels, outChannels, filterSize, stride, padding, &wInit, &bInit));
    model.append(new BatchNorm2d(16, 1e-11, 0.8));
    model.append(new ReLU());
    model.append(new MaxPool2d(2, 2));
    model.append(new Flatten());
    model.append(new Linear(outChannels*7*7, 32, &wInit, &bInit));
    model.append(new ReLU());
    model.append(new Linear(32, classes, &wInit, &bInit));
    model.append(new SoftMax());

    auto trainset = ImgDataset("./data/trainset", 1, (unsigned)1);
    auto valset = ImgDataset("./data/trainset", 1, (unsigned)2);

    trainset.resize(12);
    valset.resize(2);

    auto trainLoader = torch::data::make_data_loader(std::move(trainset.map(torch::data::transforms::Stack<>())),
                                                     torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));

    auto valLoader = torch::data::make_data_loader(std::move(valset.map(torch::data::transforms::Stack<>())),
                                                   torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));


    auto trainer = Trainer(model, &loss, batchSize);

    auto [x, y] = trainer.fit<>(*trainLoader, *valLoader, 3);
}