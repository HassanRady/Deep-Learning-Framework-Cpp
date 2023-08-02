#include "torch/torch.h"
#include "vector"
#include "tuple"
#include "iostream"
#include "memory"

// #include "Dataset.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "Model.hpp"

namespace DeepStorm
{
    class Trainer
    {
    public:
        Trainer(std::shared_ptr<Model> model, std::shared_ptr<Loss> loss, int batchSize);

        std::tuple<float, torch::Tensor> trainBatch(torch::Tensor &x, torch::Tensor &y);

        std::tuple<float, torch::Tensor> valBatch(torch::Tensor &x, torch::Tensor &y);

        // template<typename DataLoader>
        // std::tuple<float, std::vector<torch::Tensor>> trainEpoch(DataLoader& loader);

        // template<typename DataLoader>
        // std::tuple<float, std::vector<torch::Tensor>> valEpoch(DataLoader& loader);

        // template<typename DataLoader>
        // std::tuple<std::vector<float>, std::vector<float>> fit(DataLoader& trainLoader, DataLoader& valLoader, int epochs);

        template <typename DataLoader>
        std::tuple<float, std::vector<torch::Tensor>> trainEpoch(DataLoader &loader)
        {
            model->train();

            std::vector<torch::Tensor> runningPreds;
            float runningLoss = 0.0;
            int size = 0;

            torch::Tensor x, y;
            for (auto &batch : loader)
            {
                x = batch.data;
                x = x.reshape({batchSize, 28*28});
                x = x.to(torch::kCUDA).to(torch::kFloat);

                y = batch.target.to(torch::kCUDA);
                size += y.sizes()[0];

                auto [batchLoss, preds] = trainBatch(x, y);

                runningLoss += batchLoss;
                runningPreds.push_back(preds);
            }

            float epochLoss = runningLoss / size;

            std::cout << "Train loss: " << epochLoss << "\n";
            return {epochLoss, runningPreds};
        }

        template <typename DataLoader>
        std::tuple<float, std::vector<torch::Tensor>> valEpoch(DataLoader &loader)
        {
            model->eval();

            std::vector<torch::Tensor> runningPreds;
            float runningLoss = 0.0;
            int size = 0;

            torch::Tensor x, y;
            for (auto &batch : loader)
            {
                size += y.sizes()[0];

                x = batch.data.to(torch::kCUDA);
                y = batch.target.to(torch::kCUDA);

                auto [batchLoss, preds] = valBatch(x, y);

                runningLoss += batchLoss;
                runningPreds.push_back(preds);
            }

            float epochLoss = runningLoss / size;

            std::cout << "Val loss: " << epochLoss << "\n";

            return {epochLoss, runningPreds};
        }

        template <typename DataLoader>
        std::tuple<std::vector<float>, std::vector<float>> fit(DataLoader &trainLoader, DataLoader &valLoader, int epochs)
        {
            std::vector<float> trainLosses;
            std::vector<torch::Tensor> trainPreds;
            std::vector<float> valLosses;
            std::vector<torch::Tensor> valPreds;

            for (int i = 1; i <= epochs; ++i)
            {
                std::cout << "Epoch: " << i << "\n";

                auto [trainLoss, trainPred] = trainEpoch(trainLoader);

                // auto [valLoss, valPred] = valEpoch(valLoader);

                trainLosses.push_back(trainLoss);
                // valLosses.push_back(valLoss);
                //            trainPreds.insert(trainPreds);
                //            valPreds.insert(valPreds);

                // TODO metrics

                std::cout;
            }

            return {trainLosses, valLosses};
        }

    private:
        std::shared_ptr<Model> model;
        std::shared_ptr<Loss> loss;
        int batchSize;
    };
} // namespace DeepStorm
