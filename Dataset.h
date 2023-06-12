#include "torch/torch.h"

class Dataset : public torch::data::Dataset<Dataset> {
public:
    Dataset() {}

    torch::data::Example<> get(size_t index) override {
        torch::Tensor input = torch::ones({3, 32, 32}, torch::kFloat32);
        torch::Tensor target = torch::ones({1}, torch::kInt64);

        return {input, target};
    }

    torch::optional <size_t> size() const override {
        return 1;
    }

};