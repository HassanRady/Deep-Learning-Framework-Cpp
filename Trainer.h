#include "torch/torch.h"
#include "Network.h"
#include "Dataset.h"
#include "Base.h"
#include "vector"
#include "tuple"
#include "iostream"


class Trainer{
public:
    Trainer(){}
    Trainer(Network network, Dataset trainData, Dataset valData, BaseLayer loss, int batchSize): network(network), loss(loss), trainData(trainData), valData(valData), batchSize(batchSize){}

    std::tuple <torch::Tensor, torch::Tensor> trainStep(torch::Tensor& x, torch::Tensor& y) {
        torch::Tensor output = network.forward(x);
        torch::Tensor loss = this->loss.forward(output, y);

        y = this->loss.backward(y);
        network.backward(y);

        return {loss, output};
    }

    std::tuple <torch::Tensor, torch::Tensor> valStep(torch::Tensor& x, torch::Tensor& y) {
        torch::Tensor output = network.forward(x);
        torch::Tensor loss = this->loss.forward(output, y);
        return {loss, output};
    }

    std::tuple<float, std::vector<torch::Tensor>> trainEpoch() {
        network.train();

        std::vector <torch::Tensor> runningPreds;
        float runningLoss = 0.0;

        // TODO batcher
        torch::Tensor x, y;
        auto trainLoader = toDataLoader(trainData);
        for (auto &batch: trainLoader) {
            x = batch.data()->data;
            y = batch.data()->target;

            auto {batchLoss, preds} = trainStep(x, y);

            runningLoss += batchLoss;
            runningPreds.push_back(preds);
        }

        float epochLoss = runningLoss/(int)trainData.size();

        std::cout << "Train loss: " << epochLoss << "\n";

        return {epochLoss, runningPreds};
    }

    std::tuple<float, std::vector<torch::Tensor>> valEpoch() {
            network.eval();

        std::vector <torch::Tensor> runningPreds;
        float runningLoss = 0.0;

        // TODO batcher
        torch::Tensor x, y;
        auto trainLoader = toDataLoader(trainData);
        for (auto &batch: trainLoader) {
            x = batch.data()->data;
            y = batch.data()->target;

            auto {batchLoss, preds} = valStep(x, y);

            runningLoss += batchLoss;
            runningPreds.push_back(preds);
        }

        float epochLoss = runningLoss/valData.size();

        std::cout << "Train loss: " << epochLoss << "\n";

        return {epochLoss, runningPreds};
    }

    auto toDataLoader(Dataset dataset) {
        auto dataloader = torch::data::make_data_loader(
                std::move(dataset),
                torch::data::DataLoaderOptions().batch_size(batchSize)
        );
        return dataloader;
    }

    std::tuple<std::vector<float>, std::vector<float>> fit(int epochs) {
        std::vector<float> trainLosses;
        std::vector <torch::Tensor> trainPreds;
        std::vector<float> valLosses;
        std::vector <torch::Tensor> valPreds;

        // TODO metrics

        for (int i = 1; i == epochs; ++i) {
            std::cout << "Epoch: " << i << "\n";

            auto [trainLoss, trainPred] = trainEpoch();
            auto [valLoss, valPred] = valEpoch();

            trainLosses.push_back(trainLoss);
            valLosses.push_back(valLoss);
//            trainPreds.insert(trainPreds);
//            valPreds.insert(valPreds);

            std::cout;
        }

        return {trainLosses, valLosses};
    }




private:
    Network network;
    BaseLayer loss;
    Dataset trainData;
    Dataset valData;
    int batchSize;

};