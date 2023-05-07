#include <iostream>
#include <torch/torch.h>

class WeightInitializer {
public:
    WeightInitializer(bool cuda = true) : cuda(cuda) {}

    virtual torch::Tensor initialize(torch::IntArrayRef weightShape, int fanIn, int fanOut) = 0;

protected:
    bool cuda;
};

class Constant : public WeightInitializer {
public:
    Constant(float scalar = 0.1, bool cuda = true) : scalar(scalar), WeightInitializer(cuda) {
    }

    torch::Tensor initialize(torch::IntArrayRef weightShape, int fanIn, int fanOut) override {
        if (cuda)
            return torch::full({fanIn, fanOut}, scalar, torch::kCUDA);
        else
            return torch::full({fanIn, fanOut}, scalar);
    }

private:
    float scalar;
};

class UniformRandom : public WeightInitializer {
public:
    UniformRandom(bool cuda = true) : WeightInitializer(cuda) {}

    torch::Tensor initialize(torch::IntArrayRef weightShape, int fanIn, int fanOut) override {
        if (cuda)
            return torch::rand(weightShape, torch::kCUDA);
        else
            return torch::rand(weightShape);
    }
};

class Xavier : public WeightInitializer {
public:
    Xavier(bool cuda = true) : WeightInitializer(cuda) {}

    torch::Tensor initialize(torch::IntArrayRef weightShape, int fanIn, int fanOut) override {
        auto sigma = std::sqrt((2) / (fanIn + fanOut));
        if (cuda)
            return torch::nn::init::normal_(torch::empty(weightShape), 0, sigma, torch::kCUDA);
        else
            return torch::nn::init::normal_(torch::empty(weightShape), 0, sigma);
    }
};

class He : public WeightInitializer {
public:
    He(bool cuda = true) : WeightInitializer(cuda) {}

    torch::Tensor initialize(torch::IntArrayRef weightShape, int fanIn, int fanOut) override {
        auto sigma = std::sqrt((2) / (fanIn + fanOut));
        if (cuda)
            return torch::nn::init::normal_(torch::empty(weightShape), 0, sigma, torch::kCUDA);
        else
            return torch::nn::init::normal_(torch::empty(weightShape), 0, sigma);
    }
};