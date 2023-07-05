#include "ImgDataset.h"

using namespace DeepStorm::Datasets;

ImgDataset::ImgDataset(std::string path, int channels = 3, unsigned seed = 42)
{
    auto [classes, examples] = readDatasetDir(path);
    this->classes = classes;

    auto [xs, ys] = shuffle(examples, (unsigned)seed);

    images = process_images(xs, channels);
    labels = process_labels(ys);
};

struct Example
{
    std::string x, y;
};

std::vector<std::string> ImgDataset::readImgDir(std::string path)
{
    vector<string> imgs;
    for (const auto &entry : std::filesystem::directory_iterator(path))
        imgs.push_back(entry.path());
    return imgs;
}

std::tuple<std::vector<string>, std::vector<Example>> ImgDataset::readDatasetDir(std::string path)
{
    vector<string> classes, imgs, labels;
    std::vector<Example> examples;
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        classes.push_back(entry.path().filename());
        for (const auto &file : std::filesystem::directory_iterator(entry))
        {
            Example example;
            example.x = file.path();
            example.y = entry.path().filename();
            examples.push_back(example);
        }
    }
    return {classes, examples};
}

torch::Tensor ImgDataset::readData(std::string loc, int channels)
{
    cv::Mat img = cv::imread(loc);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, channels}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});

    return img_tensor.clone();
}

torch::Tensor ImgDataset::read_label(int label)
{
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

vector<torch::Tensor> ImgDataset::process_images(vector<string> list_images, int channels)
{
    vector<torch::Tensor> states;
    for (std::vector<string>::iterator it = list_images.begin(); it != list_images.end(); ++it)
    {
        torch::Tensor img = readData(*it, channels);
        states.push_back(img);
    }
    return states;
}

vector<torch::Tensor> ImgDataset::process_labels(vector<string> list_labels)
{
    vector<torch::Tensor> labels;
    for (std::vector<string>::iterator it = list_labels.begin(); it != list_labels.end(); ++it)
    {
        torch::Tensor label = read_label(stoi(*it));
        labels.push_back(label);
    }
    return labels;
}

torch::Tensor ImgDataset::toOneHotEncoding(torch::Tensor &labels, int numClasses)
{
    torch::Tensor identity = torch::eye(numClasses);
    torch::Tensor oneHot = identity.index_select(0, labels);
    return oneHot.squeeze_();
}

std::tuple<std::vector<std::string>, std::vector<std::string>> ImgDataset::shuffle(std::vector<Example> &examples, unsigned seed)
{
    std::srand(seed);
    std::random_shuffle(examples.begin(), examples.end());

    std::vector<std::string> xs, ys;

    for (int i = 0; i < examples.size(); ++i)
    {
        xs.push_back(examples[i].x);
        ys.push_back(examples[i].y);
    }
    return {xs, ys};
}

torch::data::Example<> ImgDataset::get(size_t index) 
{
    torch::Tensor sampleImg = images.at(index);
    torch::Tensor sample_label = labels.at(index);
    torch::Tensor oneHotLabel = toOneHotEncoding(sample_label, classes.size());
    return {sampleImg.clone(), oneHotLabel.clone()};
};

torch::optional<size_t> ImgDataset::size() const 
{
    return labels.size();
};

void ImgDataset::resize(int size)
{
    if (size <= labels.size())
    {
        images = std::vector<torch::Tensor>(images.begin(), images.begin() + size);
        labels = std::vector<torch::Tensor>(labels.begin(), labels.begin() + size);
    }
}