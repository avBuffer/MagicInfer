#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../include/layer/details/linear.hpp"

using namespace magic_infer;


TEST(test_layer, forward_linear1) 
{
    const uint32_t in_features = 32;
    const uint32_t out_features = 64;
    const uint32_t in_dims = 1280;

    LinearLayer linear_layer(in_features, out_features, false);
    vector<float> weights(in_features * out_features, 1.f);
    linear_layer.set_weights(weights);

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    input->Fill(1.f);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output_tensor = outputs.at(i);
        for (int j = 0; j < output_tensor->size(); ++j) {
            ASSERT_EQ(output_tensor->index(j), in_features);
        }
    }
}


TEST(test_layer, forward_linear2) 
{
    const uint32_t in_features = 32;
    const uint32_t out_features = 64;
    const uint32_t in_dims = 1280;

    LinearLayer linear_layer(in_features, out_features, false);
    vector<float> weights(in_features * out_features, 1.f);
    linear_layer.set_weights(weights);

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    input->Fill(2.f);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output_tensor = outputs.at(i);
        for (int j = 0; j < output_tensor->size(); ++j) {
            ASSERT_EQ(output_tensor->index(j), in_features * 2);
        }
    }
}



TEST(test_layer, forward_linear3) 
{
    const uint32_t in_features = 8;
    const uint32_t out_features = 12;
    const uint32_t in_dims = 4;
    
    LinearLayer linear_layer(in_features, out_features, false);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    
    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(float(j + 1));
        }
    }

    linear_layer.set_weights(weights_raw);
    input->Fill(1.f);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output_tensor = outputs.at(i);
        for (int j = 0; j < output_tensor->size(); ++j) {
            ASSERT_EQ(output_tensor->index(j), 36);
        }
    }
}


TEST(test_layer, forward_linear4) 
{
    const uint32_t in_features = 64;
    const uint32_t out_features = 128;
    const uint32_t in_dims = 4;
    
    LinearLayer linear_layer(in_features, out_features, false);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    
    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(float(j + 1));
        }
    }

    linear_layer.set_weights(weights_raw);
    input->Fill(1.f);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output_tensor = outputs.at(i);
        for (int j = 0; j < output_tensor->size(); ++j) {
            ASSERT_EQ(output_tensor->index(j), 2080 * 1.f);
        }
    }
}


TEST(test_layer, forward_linear5) 
{
    const uint32_t in_features = 64;
    const uint32_t out_features = 128;
    const uint32_t in_dims = 4;
    
    LinearLayer linear_layer(in_features, out_features, false);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    
    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(float(j + 1));
        }
    }

    linear_layer.set_weights(weights_raw);
    input->Fill(2.f);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output_tensor = outputs.at(i);
        for (int j = 0; j < output_tensor->size(); ++j) {
            ASSERT_EQ(output_tensor->index(j), 2080 * 2.f);
        }
    }
}


TEST(test_layer, forward_linear6) 
{
    const uint32_t in_features = 2;
    const uint32_t out_features = 4;
    const uint32_t in_dims = 3;

    LinearLayer linear_layer(in_features, out_features, false);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);

    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(float(j + 1));
        }
    }

    linear_layer.set_weights(weights_raw);
    input->Fill({1, 2, 3, 4, 5, 6});

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    const auto &result = outputs.front();
    for (int i = 0; i < out_features; ++i) {
        ASSERT_EQ(result->at(0, i, 0), 9);
        ASSERT_EQ(result->at(0, i, 1), 12);
        ASSERT_EQ(result->at(0, i, 2), 15);
    }
}


TEST(test_layer, forward_linear7) 
{
    const uint32_t in_features = 3;
    const uint32_t out_features = 4;
    const uint32_t in_dims = 3;

    LinearLayer linear_layer(in_features, out_features, false);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    
    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(float(j + 1));
        }
    }

    linear_layer.set_weights(weights_raw);
    input->Fill({1, 2, 3, 4, 5, 6, 7, 8, 9});

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    const auto &result = outputs.front();
    for (int i = 0; i < out_features; ++i) {
        ASSERT_EQ(result->at(0, i, 0), 30);
        ASSERT_EQ(result->at(0, i, 1), 36);
        ASSERT_EQ(result->at(0, i, 2), 42);
    }
}


TEST(test_layer, forward_linear8) 
{
    const uint32_t in_features = 3;
    const uint32_t out_features = 5;
    const uint32_t in_dims = 4;

    LinearLayer linear_layer(in_features, out_features, false);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);

    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(float(j + 1));
        }
    }

    linear_layer.set_weights(weights_raw);
    input->Fill({1, 2, 3, 13, 4, 5, 6, 15, 7, 8, 9, 16});

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);

    const auto &result = outputs.front();
    for (int i = 0; i < out_features; ++i) {
        ASSERT_EQ(result->at(0, i, 0), 30);
        ASSERT_EQ(result->at(0, i, 1), 36);
        ASSERT_EQ(result->at(0, i, 2), 42);
        ASSERT_EQ(result->at(0, i, 3), 91);
    }
}


TEST(test_layer, forward_linear9) 
{
    const uint32_t in_features = 32;
    const uint32_t out_features = 48;
    const uint32_t in_dims = 4;
    LinearLayer linear_layer(in_features, out_features, false);
    
    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(1.f);
        }
    }

    linear_layer.set_weights(weights_raw);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    input->Fill(1.f);

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    const auto &result = outputs.front();
    for (int i = 0; i < out_features; ++i) {
        ASSERT_EQ(result->at(0, i, 0), 32);
        ASSERT_EQ(result->at(0, i, 1), 32);
        ASSERT_EQ(result->at(0, i, 2), 32);
        ASSERT_EQ(result->at(0, i, 3), 32);
    }
}


TEST(test_layer, forward_linear10) 
{
    const uint32_t in_features = 32;
    const uint32_t out_features = 48;
    const uint32_t in_dims = 4;
    LinearLayer linear_layer(in_features, out_features, false);

    vector<float> weights_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(1.f);
        }
    }

    linear_layer.set_weights(weights_raw);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    input->Fill(1.f);

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    
    const auto &result = outputs.front();
    for (int i = 0; i < out_features; ++i) {
        ASSERT_EQ(result->at(0, i, 0), 32);
        ASSERT_EQ(result->at(0, i, 1), 32);
        ASSERT_EQ(result->at(0, i, 2), 32);
        ASSERT_EQ(result->at(0, i, 3), 32);
    }
}


TEST(test_layer, forward_linear11) 
{
    const uint32_t in_features = 32;
    const uint32_t out_features = 48;
    const uint32_t in_dims = 4;
    LinearLayer linear_layer(in_features, out_features, false);

    vector<float> weights_raw;
    vector<float> input_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(1.f);
        }
    }

    float index = 1.f;
    for (int i = 0; i < in_features; ++i) {
        for (int j = 0; j < in_dims; ++j) {
            input_raw.push_back(index);
            index += 1;
        }
    }

    linear_layer.set_weights(weights_raw);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    input->Fill(input_raw);

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);

    const auto &result = outputs.front();
    for (int i = 0; i < out_features; ++i) {
        ASSERT_EQ(result->at(0, i, 0), 2016);
        ASSERT_EQ(result->at(0, i, 1), 2048);
        ASSERT_EQ(result->at(0, i, 2), 2080);
        ASSERT_EQ(result->at(0, i, 3), 2112);
    }
}


TEST(test_layer, forward_linear12) 
{
    const uint32_t in_features = 32;
    const uint32_t out_features = 96;
    const uint32_t in_dims = 5;
    LinearLayer linear_layer(in_features, out_features, false);

    vector<float> weights_raw;
    vector<float> input_raw;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            weights_raw.push_back(1.f);
        }
    }

    float index = 1.f;
    for (int i = 0; i < in_features; ++i) {
        for (int j = 0; j < in_dims; ++j) {
            input_raw.push_back(index);
            index += 1;
        }
    }

    linear_layer.set_weights(weights_raw);
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, in_features, in_dims);
    input->Fill(input_raw);

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    shared_ptr<Tensor<float>> output = make_shared<Tensor<float>>(1, out_features, in_dims);
    vector<shared_ptr<Tensor<float>>> outputs;
    outputs.push_back(output);

    const auto status = linear_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);

    const auto &result = outputs.front();
    for (int i = 0; i < out_features; ++i) {
        ASSERT_EQ(result->at(0, i, 0), 2512);
        ASSERT_EQ(result->at(0, i, 1), 2544);
        ASSERT_EQ(result->at(0, i, 2), 2576);
        ASSERT_EQ(result->at(0, i, 3), 2608);
        ASSERT_EQ(result->at(0, i, 4), 2640);
    }
}
