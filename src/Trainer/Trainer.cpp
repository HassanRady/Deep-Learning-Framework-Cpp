#include "Trainer.hpp"

using namespace DeepStorm;

template <class Dataset>
Trainer<Dataset>::Trainer(Model model, Dataset &trainData, Dataset &valData, std::unique_ptr<DeepStorm::Loss> loss, int batchSize)
{
    Trainer<Dataset>::model = model;
    Trainer<Dataset>::loss = std::make_unique<Loss>(loss);
    Trainer<Dataset>::trainData = trainData;
    Trainer<Dataset>::valData = valData;
    Trainer<Dataset>::batchSize = batchSize;
}

template <class Dataset>
std::tuple<float, torch::Tensor> Trainer<Dataset>::trainBatch(torch::Tensor &x, torch::Tensor &y)
{
    torch::Tensor output = Trainer<Dataset>::model.forward(x);
    float loss = Trainer<Dataset>::loss->forward(output, y);

    y = Trainer<Dataset>::loss->backward(y);
    Trainer<Dataset>::model.backward(y);

    return {loss, output};
}

template <class Dataset>
std::tuple<float, torch::Tensor> Trainer<Dataset>::valBatch(torch::Tensor &x, torch::Tensor &y)
{
    torch::Tensor output = Trainer<Dataset>::model.forward(x);
    float loss = Trainer<Dataset>::loss->forward(output, y);
    return {loss, output};
}

template <class Dataset>
std::tuple<float, std::vector<torch::Tensor>> Trainer<Dataset>::trainEpoch()
{
    Trainer<Dataset>::model.train();

    std::vector<torch::Tensor> runningPreds;
    float runningLoss = 0.0;

    torch::Tensor x, y;
    for (auto &batch : *trainData)
    {
        x = batch.data.to(torch::kCUDA);
        y = batch.target.to(torch::kCUDA);

        auto [batchLoss, preds] = trainBatch(x, y);

        runningLoss += batchLoss;
        runningPreds.push_back(preds);
    }

    float epochLoss = runningLoss / trainData->size().value();

    std::cout << "Train loss: " << epochLoss << "\n";

    return {epochLoss, runningPreds};
}

template <class Dataset>
std::tuple<float, std::vector<torch::Tensor>> Trainer<Dataset>::valEpoch()
{
    Trainer<Dataset>::model.eval();

    std::vector<torch::Tensor> runningPreds;
    float runningLoss = 0.0;

    torch::Tensor x, y;
    for (auto &batch : *valData)
    {
        x = batch.data.to(torch::kCUDA);
        y = batch.target.to(torch::kCUDA);

        auto [batchLoss, preds] = Trainer<Dataset>::valBatch(x, y);

        runningLoss += batchLoss;
        runningPreds.push_back(preds);
    }

    float epochLoss = runningLoss / valData->size().value();

    std::cout << "Val loss: " << epochLoss << "\n";

    return {epochLoss, runningPreds};
}

template <class Dataset>
std::tuple<std::vector<float>, std::vector<float>> Trainer<Dataset>::fit(int epochs)
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