#pragma once

#include <iostream>
#include <torch/torch.h>

class WeightInitializer {
public:
    WeightInitializer() {}

    virtual int initialize(torch::Tensor &tensor, int fanIn, int fanOut) {}
};

class Constant : public WeightInitializer {
public:
    Constant(float scalar = 0.1) : scalar(scalar) {
    }

    int initialize(torch::Tensor &tensor, int fanIn, int fanOut) override {
        tensor.fill_(scalar);
        return 0;
    }

private:
    float scalar;
};

class UniformRandom : public WeightInitializer {
public:
    UniformRandom() {}

     int initialize(torch::Tensor &tensor, int fanIn, int fanOut) override {
             tensor.fill_(torch::rand_like(tensor));
        return 0;
    }
};

class Xavier : public WeightInitializer {
public:
    Xavier() {}

     int initialize(torch::Tensor &tensor, int fanIn, int fanOut) override {
        auto sigma = std::sqrt((float)(2) / (fanIn + fanOut));
        torch::nn::init::normal_(tensor, 0, sigma);
        return 0;
    }
};

class He : public WeightInitializer {
public:
    He() {}

    int initialize(torch::Tensor &tensor, int fanIn, int fanOut) override {
        auto sigma = std::sqrt((float) (2) / (fanIn + fanOut));
        torch::nn::init::normal_(tensor, 0, sigma);
        return 0;
    }
};