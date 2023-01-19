#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../include/layer/details/flatten.hpp"

using namespace magic_infer;


TEST(test_layer, forward_flatten_layer1) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(8, 24, 32);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    vector<shared_ptr<Tensor<float>>> outputs(1);
    inputs.push_back(input);

    FlattenLayer flatten_layer(1, 3);
    const auto status = flatten_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);

    const auto &shapes = outputs.front()->shapes();
    ASSERT_EQ(shapes.size(), 3);

    ASSERT_EQ(shapes.at(0), 1);
    ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
    ASSERT_EQ(shapes.at(2), 1);

    const auto &raw_shapes = outputs.front()->raw_shapes();
    ASSERT_EQ(raw_shapes.size(), 1);
    ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

    uint32_t size1 = inputs.front()->size();
    uint32_t size2 = outputs.front()->size();
    ASSERT_EQ(size1, size2);

    for (uint32_t i = 0; i < size1; ++i) {
        ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
    }
}


TEST(test_layer, forward_flatten_layer2) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(8, 24, 32);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    vector<shared_ptr<Tensor<float>>> outputs(1);
    inputs.push_back(input);

    FlattenLayer flatten_layer(1, -1); // 1 3 -> 0 2
    const auto status = flatten_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);

    const auto &shapes = outputs.front()->shapes();
    ASSERT_EQ(shapes.size(), 3);

    ASSERT_EQ(shapes.at(0), 1);
    ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
    ASSERT_EQ(shapes.at(2), 1);

    const auto &raw_shapes = outputs.front()->raw_shapes();
    ASSERT_EQ(raw_shapes.size(), 1);

    ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

    uint32_t size1 = inputs.front()->size();
    uint32_t size2 = outputs.front()->size();
    ASSERT_EQ(size1, size2);

    for (uint32_t i = 0; i < size1; ++i) {
        ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
    }
}


TEST(test_layer, forward_flatten_layer3) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(8, 24, 32);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    vector<shared_ptr<Tensor<float>>> outputs(1);
    inputs.push_back(input);

    FlattenLayer flatten_layer(1, 2); // 0 1
    const auto status = flatten_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);;
    ASSERT_EQ(outputs.size(), 1);

    const auto &shapes = outputs.front()->shapes();
    ASSERT_EQ(shapes.size(), 3);

    ASSERT_EQ(shapes.at(0), 1);
    ASSERT_EQ(shapes.at(1), 8 * 24);
    ASSERT_EQ(shapes.at(2), 32);

    const auto &raw_shapes = outputs.front()->raw_shapes();
    ASSERT_EQ(raw_shapes.size(), 2);

    ASSERT_EQ(raw_shapes.at(0), 8 * 24);
    ASSERT_EQ(raw_shapes.at(1), 32);

    uint32_t size1 = inputs.front()->size();
    uint32_t size2 = outputs.front()->size();
    ASSERT_EQ(size1, size2);

    for (uint32_t i = 0; i < size1; ++i) {
        ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
    }
}


TEST(test_layer, forward_flatten_layer4) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(8, 24, 32);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    vector<shared_ptr<Tensor<float>>> outputs(1);
    inputs.push_back(input);

    FlattenLayer flatten_layer(1, -2); // 0 1
    const auto status = flatten_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    ASSERT_EQ(outputs.size(), 1);

    const auto &shapes = outputs.front()->shapes();
    ASSERT_EQ(shapes.size(), 3);

    ASSERT_EQ(shapes.at(0), 1);
    ASSERT_EQ(shapes.at(1), 8 * 24);
    ASSERT_EQ(shapes.at(2), 32);

    const auto &raw_shapes = outputs.front()->raw_shapes();
    ASSERT_EQ(raw_shapes.size(), 2);

    ASSERT_EQ(raw_shapes.at(0), 8 * 24);
    ASSERT_EQ(raw_shapes.at(1), 32);

    uint32_t size1 = inputs.front()->size();
    uint32_t size2 = outputs.front()->size();
    ASSERT_EQ(size1, size2);

    for (uint32_t i = 0; i < size1; ++i) {
        ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
    }
}


TEST(test_layer, forward_flatten_layer5) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(8, 24, 32);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    vector<shared_ptr<Tensor<float>>> outputs(1);
    inputs.push_back(input);

    FlattenLayer flatten_layer(2, 3); // 1 2
    const auto status = flatten_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    ASSERT_EQ(outputs.size(), 1);

    const auto &shapes = outputs.front()->shapes();
    ASSERT_EQ(shapes.size(), 3);

    ASSERT_EQ(shapes.at(0), 1);
    ASSERT_EQ(shapes.at(1), 8);
    ASSERT_EQ(shapes.at(2), 24 * 32);

    const auto &raw_shapes = outputs.front()->raw_shapes();
    ASSERT_EQ(raw_shapes.size(), 2);

    ASSERT_EQ(raw_shapes.at(0), 8);
    ASSERT_EQ(raw_shapes.at(1), 24 * 32);

    uint32_t size1 = inputs.front()->size();
    uint32_t size2 = outputs.front()->size();
    ASSERT_EQ(size1, size2);

    for (uint32_t i = 0; i < size1; ++i) {
        ASSERT_EQ(inputs.front()->index(i), outputs.front()->index(i));
    }
}
