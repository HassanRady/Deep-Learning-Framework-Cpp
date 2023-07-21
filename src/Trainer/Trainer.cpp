#include "Trainer.hpp"

using namespace DeepStorm;


Trainer::Trainer(std::unique_ptr<Model> model, Loss* loss, int batchSize)
{
    Trainer::model = std::move(model);
    // Trainer::loss = std::make_unique<Loss>(loss);
    Trainer::loss = loss;
    Trainer::batchSize = batchSize;
}


std::tuple<float, torch::Tensor> Trainer::trainBatch(torch::Tensor &x, torch::Tensor &y)
{
    torch::Tensor output = Trainer::model->forward(x);
    float loss = Trainer::loss->forward(output, y);

    y = Trainer::loss->backward(y);
    Trainer::model->backward(y);

    return {loss, output};
}


std::tuple<float, torch::Tensor> Trainer::valBatch(torch::Tensor &x, torch::Tensor &y)
{
    torch::Tensor output = Trainer::model->forward(x);
    float loss = Trainer::loss->forward(output, y);
    return {loss, output};
}
