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
        class ImgDataset : public torch::data::datasets::Dataset<ImgDataset>
        {
        public:
            struct Example
            {
                std::string x, y;
            };

            ImgDataset(std::string path, int channels, unsigned seed);

            torch::data::Example<> get(size_t index) override;

            torch::optional<size_t> size() const override;

            torch::Tensor toOneHotEncoding(torch::Tensor &labels, int numClasses);

            void resize(int size);

            std::vector<std::string> readImgDir(std::string path);

            std::tuple<std::vector<std::string>, std::vector<Example>> readDatasetDir(std::string path);

            torch::Tensor readData(std::string loc, int channels);

            torch::Tensor read_label(int label);

            std::vector<torch::Tensor> process_images(std::vector<std::string> list_images, int channels);

            std::vector<torch::Tensor> process_labels(std::vector<std::string> list_labels);


            std::tuple<std::vector<std::string>, std::vector<std::string>> shuffle(std::vector<Example> &examples, unsigned seed);

            std::vector<torch::Tensor> xs, ys;
            std::vector<std::string> classes;
        };
    } // namespace Datasets

} // namespace DeepStorm
