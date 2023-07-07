#pragma once

#include "torch/torch.h"

namespace DeepStorm
{
    class Dataset : public torch::data::datasets::Dataset<Dataset>
    {
    public:
        Dataset();

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

        torch::Tensor toOneHotEncoding(torch::Tensor &labels, int numClasses);
    };
} // namespace DeepStorm
