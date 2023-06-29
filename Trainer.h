#include "torch/torch.h"
#include "Network.h"
#include "Dataset.h"
#include "Base.h"
#include "vector"
#include "tuple"
#include "iostream"
#include "Loss.h"


using mappedDataset = torch::data::datasets::MapDataset<Dataset, torch::data::transforms::Stack<torch::data::Example<>>>;

class Trainer {
public:
    Trainer(Network network, mappedDataset trainData, mappedDataset valData, Loss* loss, int batchSize) : network(network),
                                                                                                  loss(loss),
                                                                                                  trainData(trainData),
                                                                                                  valData(valData),
                                                                                                  batchSize(
                                                                                                          batchSize) {}

    std::tuple<float, torch::Tensor> trainStep(torch::Tensor &x, torch::Tensor &y) {
        torch::Tensor output = network.forward(x);
        float loss = this->loss->forward(output, y);

        y = this->loss->backward(y);
        network.backward(y);

        return {loss, output};
    }

    std::tuple<float, torch::Tensor> valStep(torch::Tensor &x, torch::Tensor &y) {
        torch::Tensor output = network.forward(x);
        float loss = this->loss->forward(output, y);
        return {loss, output};
    }

    std::tuple<float, std::vector<torch::Tensor>> trainEpoch() {
        network.train();

        std::vector <torch::Tensor> runningPreds;
        float runningLoss = 0.0;

        auto trainLoader = torch::data::make_data_loader(std::move(trainData),
                                                    torch::data::DataLoaderOptions().batch_size(batchSize));
        torch::Tensor x, y;
        for (auto &batch: *trainLoader) {
            x = batch.data.to(torch::kCUDA);
            y = batch.target.to(torch::kCUDA);

            auto [batchLoss, preds] = trainStep(x, y);

            runningLoss += batchLoss;
            runningPreds.push_back(preds);
        }

        float epochLoss = runningLoss / trainData.size().value();

        std::cout << "Train loss: " << epochLoss << "\n";

        return {epochLoss, runningPreds};
    }

    std::tuple<float, std::vector<torch::Tensor>> valEpoch() {
        network.eval();

        std::vector <torch::Tensor> runningPreds;
        float runningLoss = 0.0;


        auto valLoader = torch::data::make_data_loader(std::move(valData),
                                                  torch::data::DataLoaderOptions().batch_size(batchSize));
        torch::Tensor x, y;
        for (auto &batch: *valLoader) {
            x = batch.data.to(torch::kCUDA);
            y = batch.target.to(torch::kCUDA);

            auto [batchLoss, preds] = valStep(x, y);

            runningLoss += batchLoss;
            runningPreds.push_back(preds);
        }

        float epochLoss = runningLoss / valData.size().value();

        std::cout << "Val loss: " << epochLoss << "\n";

        return {epochLoss, runningPreds};
    }

    std::tuple <std::vector<float>, std::vector<float>> fit(int epochs) {
        std::vector<float> trainLosses;
        std::vector <torch::Tensor> trainPreds;
        std::vector<float> valLosses;
        std::vector <torch::Tensor> valPreds;

        for (int i = 1; i <= epochs; ++i) {
            std::cout << "Epoch: " << i << "\n";

            auto [trainLoss, trainPred] = trainEpoch();
            auto [valLoss, valPred] = valEpoch();

            trainLosses.push_back(trainLoss);
            valLosses.push_back(valLoss);
//            trainPreds.insert(trainPreds);
//            valPreds.insert(valPreds);

            // TODO metrics

            std::cout;
        }

        return {trainLosses, valLosses};
    }


private:
    Network network;
    Loss* loss;
    mappedDataset trainData;
    mappedDataset valData;
    int batchSize;

};