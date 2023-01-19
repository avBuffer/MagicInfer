#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../include/layer/details/upsample.hpp"

using namespace magic_infer;


TEST(test_layer, forward_upsample1) 
{
    UpSampleLayer layer(2.f, 2.f);
    const uint32_t channels = 3;
    const uint32_t rows = 224;
    const uint32_t cols = 224;

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(channels, rows, cols);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(i);
        for (int c = 0; c < channels; ++c) {
            const auto &output_channel = output->at(i);
            const auto &input_channel = input->at(i);
            ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 2);
            ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 2);

            for (int r = 0; r < output_channel.n_rows; ++r) {
                for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
                    ASSERT_EQ(input_channel.at(r / 2, c_ / 2), output_channel.at(r, c_)) << r << " " << c_;
                }
            }
        }
    }
}


TEST(test_layer, forward_upsample2) 
{
    UpSampleLayer layer(2.f, 3.f);
    const uint32_t channels = 3;
    const uint32_t rows = 224;
    const uint32_t cols = 224;

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(channels, rows, cols);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(i);
        for (int c = 0; c < channels; ++c) {
            const auto &output_channel = output->at(i);
            const auto &input_channel = input->at(i);
            ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 2);
            ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 3);

            for (int r = 0; r < output_channel.n_rows; ++r) {
                for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
                    ASSERT_EQ(input_channel.at(r / 2, c_ / 3), output_channel.at(r, c_)) << r << " " << c_;
                }
            }
        }
    }
}


TEST(test_layer, forward_upsample3) 
{
    UpSampleLayer layer(3.f, 2.f);
    const uint32_t channels = 3;
    const uint32_t rows = 224;
    const uint32_t cols = 224;

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(channels, rows, cols);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(i);
        for (int c = 0; c < channels; ++c) {
            const auto &output_channel = output->at(i);
            const auto &input_channel = input->at(i);
            ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 3);
            ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 2);

            for (int r = 0; r < output_channel.n_rows; ++r) {
                for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
                    ASSERT_EQ(input_channel.at(r / 3, c_ / 2), output_channel.at(r, c_)) << r << " " << c_;
                }
            }
        }
    }
}


TEST(test_layer, forward_upsample4) 
{
    UpSampleLayer layer(3.f, 3.f);
    const uint32_t channels = 3;
    const uint32_t rows = 224;
    const uint32_t cols = 224;

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(channels, rows, cols);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(i);
        for (int c = 0; c < channels; ++c) {
            const auto &output_channel = output->at(i);
            const auto &input_channel = input->at(i);
            ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 3);
            ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 3);

            for (int r = 0; r < output_channel.n_rows; ++r) {
                for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
                    ASSERT_EQ(input_channel.at(r / 3, c_ / 3), output_channel.at(r, c_)) << r << " " << c_;
                }
            }
        }
    }
}


TEST(test_layer, forward_upsample5) 
{
    UpSampleLayer layer(4.f, 4.f);
    const uint32_t channels = 3;
    const uint32_t rows = 224;
    const uint32_t cols = 224;

    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(channels, rows, cols);
    input->Rand();

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &output = outputs.at(i);
        for (int c = 0; c < channels; ++c) {
            const auto &output_channel = output->at(i);
            const auto &input_channel = input->at(i);
            ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 4);
            ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 4);

            for (int r = 0; r < output_channel.n_rows; ++r) {
                for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
                    ASSERT_EQ(input_channel.at(r / 4, c_ / 4), output_channel.at(r, c_)) << r << " " << c_;
                }
            }
        }
    }
}
