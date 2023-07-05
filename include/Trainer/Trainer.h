#include "torch/torch.h"
#include "vector"
#include "tuple"
#include "iostream"

#include "ImgDataset.h"
#include "Layer.h"
#include "Loss.h"
#include "Model.h"

namespace DeepStorm
{
    class Trainer
    {
    public:
        Trainer(Model model, ImgDataset trainData, ImgDataset valData, Loss *loss, int batchSize);

        std::tuple<float, torch::Tensor> trainBatch(torch::Tensor &x, torch::Tensor &y);

        std::tuple<float, torch::Tensor> valBatch(torch::Tensor &x, torch::Tensor &y);

        std::tuple<float, std::vector<torch::Tensor>> trainEpoch();

        std::tuple<float, std::vector<torch::Tensor>> valEpoch();

        std::tuple<std::vector<float>, std::vector<float>> fit(int epochs);

    private:
        Network network;
        Loss *loss;
        Dataset trainData;
        Dataset valData;
        int batchSize;
    };
} // namespace DeepStorm
