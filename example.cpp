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
#include "Sigmoid.hpp"
#include "Softmax.hpp"
#include "He.hpp"
#include "Constant.hpp"
#include "Optimizer.hpp"
#include "Adam.hpp"
#include "SGD.hpp"
#include "ImgDataset.hpp"
#include "Trainer.hpp"
#include "CrossEntropy.hpp"

using namespace std;

using namespace DeepStorm;
using namespace DeepStorm::Layers;
using namespace DeepStorm::Activations;
using namespace DeepStorm::Optimizers;
using namespace DeepStorm::Initializers;
using namespace DeepStorm::Datasets;
using namespace DeepStorm::Losses;

int main()
{
    torch::manual_seed(1);

    auto inChannels = 1;
    auto filterSize = 3;
    auto outChannels = 16;
    auto stride = 1;
    auto padding = "same";
    auto classes = 10;

    std::shared_ptr<Loss> loss = std::make_shared<CrossEntropyLoss>(1e-09);
    std::shared_ptr<Model> model = std::make_shared<Model>();

    // model->append(new Conv2d(inChannels, outChannels, filterSize, stride, padding, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    // model->append(new BatchNorm2d(outChannels, new Adam(0.001, 0.9, 0.9, 1e-07), 1e-11, 0.8));
    // model->append(new Dropout(0.3));
    // model->append(new ReLU());
    // model->append(new MaxPool2d(2, 2));
    // model->append(new Conv2d(outChannels, outChannels, filterSize, stride, padding, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    // model->append(new BatchNorm2d(outChannels, new Adam(0.001, 0.9, 0.9, 1e-07), 1e-11, 0.8));
    // model->append(new ReLU());
    // model->append(new MaxPool2d(2, 2));
    // model->append(new Flatten());
    // model->append(new Linear(outChannels * 7 * 7, 128, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    // model->append(new ReLU());
    // model->append(new Linear(128, classes, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    // model->append(new ReLU());
    // model->append(new SoftMax());

    model->append(new Linear(28 * 28, 256, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    model->append(new ReLU());
    model->append(new Linear(256, 128, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    model->append(new ReLU());
    model->append(new Linear(128, 64, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    model->append(new ReLU());
    model->append(new Linear(64, 10, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
    model->append(new ReLU());
    model->append(new SoftMax());

    auto trainset = ImgDataset("./data/trainset", 1, (unsigned)9);
    auto valset = ImgDataset("./data/trainset", 1, (unsigned)2);

    // Train set original size: 42000
    // Val set original size: 18000
    trainset.resize(8000);
    valset.resize(1000);

    auto batchSize = 32;
    auto trainLoader = torch::data::make_data_loader(std::move(trainset.map(torch::data::transforms::Stack<>())),
                                                     torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));

    auto valLoader = torch::data::make_data_loader(std::move(valset.map(torch::data::transforms::Stack<>())),
                                                   torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));
    auto trainer = Trainer(model, loss, batchSize, 255.0);
    auto [x, y] = trainer.fit<>(*trainLoader, *valLoader, 50);
}