#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../include/layer/details/silu.hpp"

using namespace magic_infer;


TEST(test_layer, forward_silu1) 
{  
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 224, 512);
    input->Rand();
    vector<shared_ptr<Tensor<float>>> inputs;

    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    SiLULayer silu_layer;
    const auto status = silu_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_LE(abs(output_->index(j) - input_->index(j) / (1 + exp(-input_->index(j)))), 1e-6);
        }
    }
}


TEST(test_layer, forward_silu2) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 32, 128);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    SiLULayer silu_layer;
    const auto status = silu_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_LE(abs(output_->index(j) - input_->index(j) / (1 + exp(-input_->index(j)))), 1e-6);
        }
    }
}


TEST(test_layer, forward_silu3) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 1, 16);
    input->Rand();
    vector<shared_ptr<Tensor<float>>> inputs;

    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    SiLULayer silu_layer;
    const auto status = silu_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_LE(abs(output_->index(j) - input_->index(j) / (1 + exp(-input_->index(j)))), 1e-6);
        }
    }
}


TEST(test_layer, forward_silu4) 
{
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 320, 1);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;    
    inputs.push_back(input);
    vector<shared_ptr<Tensor<float>>> outputs(1);

    SiLULayer silu_layer;
    const auto status = silu_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < inputs.size(); ++i) {
        shared_ptr<Tensor<float>> input_ = inputs.at(i);
        shared_ptr<Tensor<float>> output_ = outputs.at(i);
        CHECK(input_->size() == output_->size());
        
        uint32_t size = input_->size();
        for (uint32_t j = 0; j < size; ++j) {
            ASSERT_LE(abs(output_->index(j) - input_->index(j) / (1 + exp(-input_->index(j)))), 1e-6);
        }
    }
}
