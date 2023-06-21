#include "iostream"
#include <torch/torch.h>

#include "Base.h"
#include "Initializers.h"
#include "Optimizer.h"


using namespace torch::indexing;


class Linear : public BaseLayer {
public:
    Linear(int inFeatures, int outFeatures, WeightInitializer weightInitializer = He(), WeightInitializer biasInitializer = Constant())
            : inFeatures(inFeatures), outFeatures(outFeatures) {
        trainable = true;
        initializable = true;

        this->weightInitializer = weightInitializer;
        this->biasInitializer = biasInitializer;

//        weights = torch::empty({inFeatures + 1, outFeatures}, torch::kCUDA);
//        bias = torch::ones({outFeatures, 1}, torch::kCUDA);
        initialize();

    }

    void initialize() {
        weights = torch::empty({inFeatures, outFeatures}, torch::kCUDA);
        bias = torch::empty({1, outFeatures}, torch::kCUDA);

        weightInitializer.initialize(weights, inFeatures, outFeatures);
        biasInitializer.initialize(bias, 1, outFeatures);

        weights = torch::cat({weights, bias}, 0);
    }

    torch::Tensor forward(torch::Tensor &inputTensor) override{
        auto inputDims = inputTensor.sizes();
        batchSize = inputDims[0];

        this->inputTensor = torch::cat({inputTensor, torch::ones({batchSize, 1}, torch::kCUDA)}, -1);
        return torch::matmul(this->inputTensor, weights);
    }

    torch::Tensor backward(torch::Tensor &errorTensor) override{

        gradientWeights = torch::matmul(inputTensor.transpose(1, 0), errorTensor);


        weights = optimizer.update(weights, gradientWeights);

        auto out = torch::matmul(errorTensor, weights.transpose(1, 0)).index({Slice(), Slice(0, inFeatures)});

        return out;
    }

public:
    int inFeatures;
    int outFeatures;

    torch::Tensor weights;
    torch::Tensor gradientWeights;

    WeightInitializer weightInitializer;
    WeightInitializer biasInitializer;

    torch::Tensor bias;
    torch::Tensor gradientBias;

    torch::Tensor inputTensor;

    int batchSize;

public:
    Optimizer optimizer;
};

