#include "iostream"
#include "vector"
#include "string"
#include "torch/torch.h"

#include "Dataset.h"
#include "Trainer.h"
#include "Optimizer.h"
#include "Loss.h"

#include "Base.h"
#include "Conv.h"
#include "Pooling.h"
#include "BatchNormalization.h"
#include "FullyConnected.h"
#include "Initializers.h"
#include "Relu.h"
#include "Dropout.h"
#include "Flatten.h"
#include "Softmax.h"
#include "Network.h"

using namespace std;

int main() {

    auto batchSize = 2;
    auto inChannels = 1;
    auto filterSize = 3;
    auto outChannels = 16;
    auto stride = 1;
    auto padding = "same";
    auto classes = 10;

    He wInit;
    Constant bInit;

    Conv2d conv1(inChannels, outChannels, filterSize, stride, padding, &wInit, &bInit);
    Conv2d conv2(outChannels, outChannels, filterSize, stride, padding, &wInit, &bInit);
    BatchNorm2d batchNorm1 (outChannels);
    BatchNorm2d batchNorm2(outChannels);
    Dropout dropout(0.3);
    MaxPool2d maxPool1(2, 2);
    MaxPool2d maxPool2(2, 2);
    ReLU relu1;
    ReLU relu2;
    ReLU relu3;
    Flatten flatten;
    Linear fc1(outChannels*7*7, 32, &wInit, &bInit);
    Linear fc2(32, classes, &wInit, &bInit);
    SoftMax softmax;

    Adam optimizer;
    CrossEntropyLoss loss;

    auto model = Network(vector<BaseLayer*>{
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
        &softmax
    });

    // model.setOptimizer(&optimizer);
    conv1.optimizer = new Adam();
    conv2.optimizer = new Adam();
    batchNorm1.optimizer = new Adam();
    batchNorm2.optimizer = new Adam();
    fc1.optimizer = new Adam();
    fc2.optimizer = new Adam();

    auto trainset = Dataset ("./data/trainset", 1, (unsigned) 1);
    auto valset = Dataset ("./data/trainset", 1);

    trainset.resize(12);
    valset.resize(2);

   auto trainer = Trainer(model, trainset, valset, &loss, batchSize);

   auto [x, y] = trainer.fit(5);

}