#include "torch/torch.h"
#include "vector"
#include "tuple"
#include "iostream"

#include "Dataset.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "Model.hpp"

namespace DeepStorm
{
    class Trainer
    {
    public:
        Trainer(Model model, DeepStorm::Dataset* trainData, DeepStorm::Dataset* valData, Loss *loss, int batchSize);

        std::tuple<float, torch::Tensor> trainBatch(torch::Tensor &x, torch::Tensor &y);

        std::tuple<float, torch::Tensor> valBatch(torch::Tensor &x, torch::Tensor &y);

        std::tuple<float, std::vector<torch::Tensor>> trainEpoch();

        std::tuple<float, std::vector<torch::Tensor>> valEpoch();

        std::tuple<std::vector<float>, std::vector<float>> fit(int epochs);

    private:
        DeepStorm::Model model;
        DeepStorm::Loss *loss;
        DeepStorm::Dataset* trainData;
        DeepStorm::Dataset* valData;
        int batchSize;
    };
} // namespace DeepStorm
