#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../include/layer/details/adaptive_avgpooling.hpp"

using namespace magic_infer;


void AveragePooling(const vector<shared_ptr<Tensor<float>>> &inputs,
    vector<shared_ptr<Tensor<float>>> &outputs, uint32_t output_h, uint32_t output_w) 
{
    const uint32_t batch = inputs.size();
    for (uint32_t i = 0; i < batch; ++i) {
        const shared_ptr<Tensor<float>> &input_data = inputs.at(i);
        const uint32_t input_h = input_data->rows();
        const uint32_t input_w = input_data->cols();
        const uint32_t input_c = input_data->channels();
        const uint32_t output_c = input_c;

        const uint32_t stride_h = uint32_t(floor(input_h / output_h));
        const uint32_t stride_w = uint32_t(floor(input_w / output_w));
        const uint32_t kernel_h = input_h - (output_h - 1) * stride_h;
        const uint32_t kernel_w = input_w - (output_w - 1) * stride_w;
        CHECK(output_w > 0 && output_h > 0);

        shared_ptr<Tensor<float>> output_data = make_shared<Tensor<float>>(output_c, output_h, output_w);
        for (uint32_t ic = 0; ic < input_c; ++ic) {
            const arma::fmat &input_channel = input_data->at(ic);
            arma::fmat &output_channel = output_data->at(ic);
            
            for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
                for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
                    const arma::fmat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
                    output_channel.at(int(r / stride_h), int(c / stride_w)) = arma::mean(arma::mean(region));
                }
            }
        }
        outputs.push_back(output_data);
    }
}


TEST(test_layer, forward_average_pooling_out1x1) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    const uint32_t input_h = 224;
    const uint32_t input_w = 224;
    const uint32_t output_h = 1;
    const uint32_t output_w = 1;
    const uint32_t input_size = 3;

    for (uint32_t i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, input_h, input_w);
        input->Rand();
        inputs.push_back(input);
    }

    vector<shared_ptr<Tensor<float>>> outputs1;
    AveragePooling(inputs, outputs1, output_h, output_w);
    ASSERT_EQ(outputs1.size(), input_size);
    AdaptiveAvgPoolingLayer average_layer(output_h, output_w);

    vector<shared_ptr<Tensor<float>>> outputs2(input_size);
    average_layer.Forward(inputs, outputs2);
    ASSERT_EQ(outputs2.size(), input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
        const auto &output1 = outputs1.at(i);
        const auto &output2 = outputs2.at(i);
        ASSERT_EQ(output1->channels(), output2->channels());
        ASSERT_EQ(output1->shapes(), output2->shapes());

        uint32_t channels = output1->channels();
        for (int c = 0; c < channels; ++c) {
            ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
        }
    }
}


TEST(test_layer, forward_average_pooling_out3x3) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    const uint32_t input_h = 224;
    const uint32_t input_w = 224;
    const uint32_t output_h = 3;
    const uint32_t output_w = 3;
    const uint32_t input_size = 3;

    for (uint32_t i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, input_h, input_w);
        input->Rand();
        inputs.push_back(input);
    }
    
    vector<shared_ptr<Tensor<float>>> outputs1;
    AveragePooling(inputs, outputs1, output_h, output_w);
    ASSERT_EQ(outputs1.size(), input_size);
    AdaptiveAvgPoolingLayer average_layer(output_h, output_w);

    vector<shared_ptr<Tensor<float>>> outputs2(input_size);
    average_layer.Forward(inputs, outputs2);
    ASSERT_EQ(outputs2.size(), input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
        const auto &output1 = outputs1.at(i);
        const auto &output2 = outputs2.at(i);
        ASSERT_EQ(output1->channels(), output2->channels());
        ASSERT_EQ(output1->shapes(), output2->shapes());
        
        uint32_t channels = output1->channels();
        for (int c = 0; c < channels; ++c) {
            ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
        }
    }
}


TEST(test_layer, forward_average_pooling_out5x5) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    const uint32_t input_h = 224;
    const uint32_t input_w = 224;
    const uint32_t output_h = 5;
    const uint32_t output_w = 5;
    const uint32_t input_size = 3;

    for (uint32_t i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, input_h, input_w);
        input->Rand();
        inputs.push_back(input);
    }
    
    vector<shared_ptr<Tensor<float>>> outputs1;
    AveragePooling(inputs, outputs1, output_h, output_w);
    ASSERT_EQ(outputs1.size(), input_size);
    AdaptiveAvgPoolingLayer average_layer(output_h, output_w);

    vector<shared_ptr<Tensor<float>>> outputs2(input_size);
    average_layer.Forward(inputs, outputs2);
    ASSERT_EQ(outputs2.size(), input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
        const auto &output1 = outputs1.at(i);
        const auto &output2 = outputs2.at(i);
        ASSERT_EQ(output1->channels(), output2->channels());
        ASSERT_EQ(output1->shapes(), output2->shapes());

        uint32_t channels = output1->channels();
        for (int c = 0; c < channels; ++c) {
            ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
        }
    }
}


TEST(test_layer, forward_average_pooling_out7x7) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    const uint32_t input_h = 224;
    const uint32_t input_w = 224;
    const uint32_t output_h = 7;
    const uint32_t output_w = 7;
    const uint32_t input_size = 3;

    for (uint32_t i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, input_h, input_w);
        input->Rand();
        inputs.push_back(input);
    }
    
    vector<shared_ptr<Tensor<float>>> outputs1;
    AveragePooling(inputs, outputs1, output_h, output_w);
    ASSERT_EQ(outputs1.size(), input_size);
    AdaptiveAvgPoolingLayer average_layer(output_h, output_w);

    vector<shared_ptr<Tensor<float>>> outputs2(input_size);
    average_layer.Forward(inputs, outputs2);
    ASSERT_EQ(outputs2.size(), input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
        const auto &output1 = outputs1.at(i);
        const auto &output2 = outputs2.at(i);
        ASSERT_EQ(output1->channels(), output2->channels());
        ASSERT_EQ(output1->shapes(), output2->shapes());
        
        uint32_t channels = output1->channels();
        for (int c = 0; c < channels; ++c) {
            ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
        }
    }
}


TEST(test_layer, forward_average_pooling_out9x9) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    const uint32_t input_h = 224;
    const uint32_t input_w = 224;
    const uint32_t output_h = 9;
    const uint32_t output_w = 9;
    const uint32_t input_size = 3;

    for (uint32_t i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, input_h, input_w);
        input->Rand();
        inputs.push_back(input);
    }
    
    vector<shared_ptr<Tensor<float>>> outputs1;
    AveragePooling(inputs, outputs1, output_h, output_w);
    ASSERT_EQ(outputs1.size(), input_size);
    AdaptiveAvgPoolingLayer average_layer(output_h, output_w);

    vector<shared_ptr<Tensor<float>>> outputs2(input_size);
    average_layer.Forward(inputs, outputs2);
    ASSERT_EQ(outputs2.size(), input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
        const auto &output1 = outputs1.at(i);
        const auto &output2 = outputs2.at(i);
        ASSERT_EQ(output1->channels(), output2->channels());
        ASSERT_EQ(output1->shapes(), output2->shapes());
        
        uint32_t channels = output1->channels();
        for (int c = 0; c < channels; ++c) {
            ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
        }
    }
}


TEST(test_layer, forward_average_pooling_out11x11) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    const uint32_t input_h = 224;
    const uint32_t input_w = 224;
    const uint32_t output_h = 11;
    const uint32_t output_w = 11;
    const uint32_t input_size = 3;

    for (uint32_t i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, input_h, input_w);
        input->Rand();
        inputs.push_back(input);
    }

    vector<shared_ptr<Tensor<float>>> outputs1;
    AveragePooling(inputs, outputs1, output_h, output_w);
    ASSERT_EQ(outputs1.size(), input_size);
    AdaptiveAvgPoolingLayer average_layer(output_h, output_w);

    vector<shared_ptr<Tensor<float>>> outputs2(input_size);
    average_layer.Forward(inputs, outputs2);
    ASSERT_EQ(outputs2.size(), input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
        const auto &output1 = outputs1.at(i);
        const auto &output2 = outputs2.at(i);
        ASSERT_EQ(output1->channels(), output2->channels());
        ASSERT_EQ(output1->shapes(), output2->shapes());
        
        uint32_t channels = output1->channels();
        for (int c = 0; c < channels; ++c) {
            ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
        }
    }
}


TEST(test_layer, forward_average_pooling_out1x11) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    const uint32_t input_h = 224;
    const uint32_t input_w = 224;
    const uint32_t output_h = 1;
    const uint32_t output_w = 11;
    const uint32_t input_size = 3;

    for (uint32_t i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, input_h, input_w);
        input->Rand();
        inputs.push_back(input);
    }

    vector<shared_ptr<Tensor<float>>> outputs1;
    AveragePooling(inputs, outputs1, output_h, output_w);
    ASSERT_EQ(outputs1.size(), input_size);
    AdaptiveAvgPoolingLayer average_layer(output_h, output_w);

    vector<shared_ptr<Tensor<float>>> outputs2(input_size);
    average_layer.Forward(inputs, outputs2);
    ASSERT_EQ(outputs2.size(), input_size);

    for (uint32_t i = 0; i < input_size; ++i) {
        const auto &output1 = outputs1.at(i);
        const auto &output2 = outputs2.at(i);
        ASSERT_EQ(output1->channels(), output2->channels());
        ASSERT_EQ(output1->shapes(), output2->shapes());
        
        uint32_t channels = output1->channels();
        for (int c = 0; c < channels; ++c) {
            ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
        }
    }
}
