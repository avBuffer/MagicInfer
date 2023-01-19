#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../include/layer/details/sigmoid.hpp"

using namespace magic_infer;


TEST(test_layer, forward_sigmoid1) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 1, 4);
    input->index(0) = 1.f;
    input->index(1) = 2.f;
    input->index(2) = 3.f;
    input->index(3) = 4.f;

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    shared_ptr<Tensor<float>> output1 = make_shared<Tensor<float>>(1, 1, 4);
    outputs.push_back(output1);

    SigmoidLayer sigmoid_layer;
    const auto status = sigmoid_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_LE(abs(outputs.front()->index(0) - 0.7310585786300049f), 1e-6);
    ASSERT_LE(abs(outputs.front()->index(1) - 0.8807970779778823f), 1e-6);
    ASSERT_LE(abs(outputs.front()->index(2) - 0.9525741268224334f), 1e-6);
    ASSERT_LE(abs(outputs.front()->index(3) - 0.9820137900379085f), 1e-6);
}


TEST(test_layer, forward_sigmoid2) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 1, 4);
    input->index(0) = 11.f;
    input->index(1) = 22.f;
    input->index(2) = 33.f;
    input->index(3) = 44.f;

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    shared_ptr<Tensor<float>> output1 = make_shared<Tensor<float>>(1, 1, 4);
    outputs.push_back(output1);

    SigmoidLayer sigmoid_layer;
    const auto status = sigmoid_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_LE(abs(outputs.front()->index(0) - 0.999983298578152f), 1e-6);
    ASSERT_LE(abs(outputs.front()->index(1) - 0.9999999997210531f), 1e-6);
    ASSERT_LE(abs(outputs.front()->index(2) - 0.9999999999999953f), 1e-6);
    ASSERT_LE(abs(outputs.front()->index(3) - 1.0f), 1e-6);
}


TEST(test_layer, forward_sigmoid3) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 224, 512);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    SigmoidLayer sigmoid_layer;
    const auto status = sigmoid_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_EQ(output_->index(j), 1.f / (1 + exp(-input_->index(j))));
        }
    }
}


TEST(test_layer, forward_sigmoid4) 
{  
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 32, 128);
    input->Rand();
    vector<shared_ptr<Tensor<float>>> inputs;

    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    SigmoidLayer sigmoid_layer;
    const auto status = sigmoid_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_EQ(output_->index(j), 1.f / (1 + exp(-input_->index(j))));
        }
    }
}


TEST(test_layer, forward_sigmoid5) 
{  
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 1, 128);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;    
    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    SigmoidLayer sigmoid_layer;
    const auto status = sigmoid_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_EQ(output_->index(j), 1.f / (1 + exp(-input_->index(j))));
        }
    }
}
