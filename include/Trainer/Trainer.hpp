#include "torch/torch.h"
#include "vector"
#include "tuple"
#include "iostream"

#include "ImgDataset.hpp"
#include "Layer.hpp"
#include "Loss.hpp"
#include "Model.hpp"

namespace DeepStorm
{
    class Trainer
    {
    public:
        Trainer(Model model, DeepStorm::Datasets::ImgDataset trainData, DeepStorm::Datasets::ImgDataset valData, Loss *loss, int batchSize);

        std::tuple<float, torch::Tensor> trainBatch(torch::Tensor &x, torch::Tensor &y);

        std::tuple<float, torch::Tensor> valBatch(torch::Tensor &x, torch::Tensor &y);

        std::tuple<float, std::vector<torch::Tensor>> trainEpoch();

        std::tuple<float, std::vector<torch::Tensor>> valEpoch();

        std::tuple<std::vector<float>, std::vector<float>> fit(int epochs);

    private:
        DeepStorm::Model network;
        DeepStorm::Loss *loss;
        DeepStorm::Datasets::ImgDataset trainData;
        DeepStorm::Datasets::ImgDataset valData;
        int batchSize;
    };
} // namespace DeepStorm
