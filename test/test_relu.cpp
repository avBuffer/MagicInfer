#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../include/layer/details/relu.hpp"

using namespace magic_infer;


TEST(test_layer, forward_relu1) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 224, 512);
    input->Rand();
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    ReluLayer relu_layer;
    const auto status = relu_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        input_->Transform([](const float f) {
            if (f < 0) return 0.f;
            else return f;
        });

        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_EQ(output_->index(j), input_->index(j));
        }
    }
}


TEST(test_layer, forward_relu2) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 32, 128);
    input->Rand();
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    ReluLayer relu_layer;
    const auto status = relu_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        input_->Transform([](const float f) {
            if (f < 0) return 0.f;
            else return f;
        });

        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_EQ(output_->index(j), input_->index(j));
        }
    }
}


TEST(test_layer, forward_relu3) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 1, 16);
    input->Rand();
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    ReluLayer relu_layer;
    const auto status = relu_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        input_->Transform([](const float f) {
            if (f < 0) return 0.f;
            else return f;
        });

        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_EQ(output_->index(j), input_->index(j));
        }
    }
}
