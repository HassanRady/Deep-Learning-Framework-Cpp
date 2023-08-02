#include "Trainer.hpp"

using namespace DeepStorm;


Trainer::Trainer(std::shared_ptr<Model> model,std::shared_ptr<Loss> loss, int batchSize, float scale)
{
    Trainer::model = model;
    Trainer::loss = loss;
    Trainer::batchSize = batchSize;
    Trainer::scale = scale;
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
