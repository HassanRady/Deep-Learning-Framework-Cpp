#include "Dataset.hpp"

using namespace DeepStorm;

Dataset::Dataset() {}

torch::data::Example<> Dataset::get(size_t index)
{
    torch::Tensor sampleImg = ImgDataset::images.at(index);
    torch::Tensor sample_label = ImgDataset::labels.at(index);
    torch::Tensor oneHotLabel = Dataset::toOneHotEncoding(sample_label, classes.size());
    return {sampleImg.clone(), oneHotLabel.clone()};
};

torch::optional<size_t> Dataset::size() const
{
    return Dataset::labels.size();
};

torch::Tensor Dataset::toOneHotEncoding(torch::Tensor &labels, int numClasses)
{
    torch::Tensor identity = torch::eye(numClasses);
    torch::Tensor oneHot = identity.index_select(0, labels);
    return oneHot.squeeze_();
}