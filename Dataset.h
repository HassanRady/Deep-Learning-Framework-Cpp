#pragma once

#include "torch/torch.h"
#include "vector"
#include "tuple"
#include "opencv4/opencv2/opencv.hpp"
#include "string"
#include "filesystem"

using namespace std;

std::vector<std::string> readImgDir(std::string path)
{
    vector<string> imgs;
    for (const auto &entry : std::filesystem::directory_iterator(path))
        imgs.push_back(entry.path());
    return imgs;
}

std::tuple<std::vector<string>, std::vector<string>, std::vector<string>> readDatasetDir(std::string path)
{
    vector<string> classes, imgs, labels;
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        classes.push_back(entry.path().filename());
        for (const auto &file : std::filesystem::directory_iterator(entry))
        {
            imgs.push_back(file.path());
            labels.push_back(entry.path().filename());
        }
    }
    return {classes, imgs, labels};
}

torch::Tensor readData(std::string loc, int channels)
{
    cv::Mat img = cv::imread(loc);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, channels}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});

    return img_tensor.clone();
}

torch::Tensor read_label(int label)
{
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

vector<torch::Tensor> process_images(vector<string> list_images, int channels)
{
    vector<torch::Tensor> states;
    for (std::vector<string>::iterator it = list_images.begin(); it != list_images.end(); ++it)
    {
        torch::Tensor img = readData(*it, channels);
        states.push_back(img);
    }
    return states;
}

vector<torch::Tensor> process_labels(vector<string> list_labels)
{
    vector<torch::Tensor> labels;
    for (std::vector<string>::iterator it = list_labels.begin(); it != list_labels.end(); ++it)
    {
        torch::Tensor label = read_label(stoi(*it));
        labels.push_back(label);
    }
    return labels;
}

torch::Tensor toOneHotEncoding(torch::Tensor& labels, int numClasses){
    torch::Tensor identity = torch::eye(numClasses);
    torch::Tensor oneHot = identity.index_select(0, labels);
    return oneHot;
}

class Dataset : public torch::data::datasets::Dataset<Dataset>
{
public:
    vector<torch::Tensor> images, labels;
    vector<string> classes;

    Dataset(std::string path, int channels=3)
    {
        auto [classes, listImages, listLabels] = readDatasetDir(path);

        this->classes = classes;

        images = process_images(listImages, channels);
        labels = process_labels(listLabels);
    };

    torch::data::Example<> get(size_t index) override
    {
        torch::Tensor sampleImg = images.at(index);
        torch::Tensor sample_label = labels.at(index);
        torch::Tensor oneHotLabel = toOneHotEncoding(sample_label, classes.size());
        return {sampleImg.clone(), oneHotLabel.clone()};
    };

    torch::optional<size_t> size() const override
    {
        return labels.size();
    };
};