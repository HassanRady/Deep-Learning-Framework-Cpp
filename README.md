# DeepStorm: Deep Learning Framework

## Summary:
Deep Learning Framework from scratch in C++ using only the tensor class from libtorch.

## Layers & DL classes in framework:
- Conv2d
- MaxPool2d
- BatchNorm2d
- Flatten
- Dropout
- Linear
- ReLU
- Softmax
- SgdWithMomentum
- Adam
- CrossEntropyLoss
- Xavier
- He

```cpp
std::shared_ptr<Loss> loss = std::make_shared<CrossEntropyLoss>(1e-09);
std::shared_ptr<Model> model = std::make_shared<Model>();

model->append(new Linear(28 * 28, 256, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
model->append(new ReLU());
model->append(new Linear(256, 128, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
model->append(new ReLU());
model->append(new Linear(128, 64, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
model->append(new ReLU());
model->append(new Linear(64, 10, new He(), new Constant(0.01), new Adam(0.001, 0.9, 0.9, 1e-07)));
model->append(new ReLU());
model->append(new SoftMax());

auto trainset = ImgDataset("./data/trainset", 1, (unsigned)9);
auto valset = ImgDataset("./data/trainset", 1, (unsigned)2);

// Train set original size: 42000
// Val set original size: 18000
trainset.resize(8000);
valset.resize(1000);

auto batchSize = 32;
auto trainLoader = torch::data::make_data_loader(std::move(trainset.map(torch::data::transforms::Stack<>())),
        torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));

auto valLoader = torch::data::make_data_loader(std::move(valset.map(torch::data::transforms::Stack<>())),
                                                torch::data::DataLoaderOptions().batch_size(batchSize).drop_last(true));
auto trainer = Trainer(model, loss, batchSize, 255.0);
auto [x, y] = trainer.fit<>(*trainLoader, *valLoader, 50);
```