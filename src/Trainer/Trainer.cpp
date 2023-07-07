#include "Trainer.hpp"

using namespace DeepStorm;

Trainer::Trainer(Model model, Trainer::ImgDataset trainData, Trainer::ImgDataset valData, Loss *loss, int batchSize)
{
    Trainer::model = model;
    Trainer::loss = loss;
    Trainer::trainData = trainData;
    Trainer::valData = valData;
    Trainer::batchSize = batchSize;
}

std::tuple<float, torch::Tensor> Trainer::trainBatch(torch::Tensor &x, torch::Tensor &y)
{
    torch::Tensor output = Trainer::model.forward(x);
    float loss = Trainer::loss->forward(output, y);

    y = Trainer::loss->backward(y);
    Trainer::model.backward(y);

    return {loss, output};
}

std::tuple<float, torch::Tensor> Trainer::valStep(torch::Tensor &x, torch::Tensor &y)
{
    torch::Tensor output = Trainer::model.forward(x);
    float loss = Trainer::loss->forward(output, y);
    return {loss, output};
}

std::tuple<float, std::vector<torch::Tensor>> Trainer::trainEpoch()
{
    Trainer::model.train();

    std::vector<torch::Tensor> runningPreds;
    float runningLoss = 0.0;

    auto trainLoader = torch::data::make_data_loader(std::move(Trainer::trainData.map(torch::data::transforms::Stack<>())),
                                                     torch::data::DataLoaderOptions().batch_size(batchSize));
    torch::Tensor x, y;
    for (auto &batch : *trainLoader)
    {
        x = batch.data.to(torch::kCUDA);
        y = batch.target.to(torch::kCUDA);

        auto [batchLoss, preds] = trainBatch(x, y);

        runningLoss += batchLoss;
        runningPreds.push_back(preds);
    }

    float epochLoss = runningLoss / trainData.size().value();

    std::cout << "Train loss: " << epochLoss << "\n";

    return {epochLoss, runningPreds};
}

std::tuple<float, std::vector<torch::Tensor>> Trainer::valEpoch()
{
    Trainer::model.eval();

    std::vector<torch::Tensor> runningPreds;
    float runningLoss = 0.0;

    auto valLoader = torch::data::make_data_loader(std::move(Trainer::valData.map(torch::data::transforms::Stack<>())),
                                                   torch::data::DataLoaderOptions().batch_size(batchSize));
    torch::Tensor x, y;
    for (auto &batch : *valLoader)
    {
        x = batch.data.to(torch::kCUDA);
        y = batch.target.to(torch::kCUDA);

        auto [batchLoss, preds] = Trainer::valStep(x, y);

        runningLoss += batchLoss;
        runningPreds.push_back(preds);
    }

    float epochLoss = runningLoss / valData.size().value();

    std::cout << "Val loss: " << epochLoss << "\n";

    return {epochLoss, runningPreds};
}

std::tuple<std::vector<float>, std::vector<float>> Trainer::fit(int epochs)
{
    std::vector<float> trainLosses;
    std::vector<torch::Tensor> trainPreds;
    std::vector<float> valLosses;
    std::vector<torch::Tensor> valPreds;

    for (int i = 1; i <= epochs; ++i)
    {
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