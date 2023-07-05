#pragma once

#include "torch/torch.h"
#include "vector"
#include "tuple"
#include "opencv4/opencv2/opencv.hpp"
#include "string"
#include "filesystem"
#include "random"

namespace DeepStorm
{
    namespace Datasets
    {
        class ImgDataset : public torch::data::datasets::Dataset<Dataset>
        {
        public:
            ImgDataset(std::string path, int channels = 3, unsigned seed = 42);

            torch::data::Example<> get(size_t index) override;

            torch::optional<size_t> size() const override;

            void resize(int size);

            std::vector<std::string> readImgDir(std::string path);

            std::tuple<std::vector<string>, std::vector<Example>> readDatasetDir(std::string path);

            torch::Tensor readData(std::string loc, int channels);

            torch::Tensor read_label(int label);

            vector<torch::Tensor> process_images(vector<string> list_images, int channels);

            vector<torch::Tensor> process_labels(vector<string> list_labels);

            torch::Tensor toOneHotEncoding(torch::Tensor &labels, int numClasses);

            std::vector<torch::Tensor> images, labels;
            std::vector<std::string> classes;
        };
    } // namespace Datasets

} // namespace DeepStorm
