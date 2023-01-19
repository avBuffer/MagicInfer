#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../include/layer/details/concat.hpp"

using namespace magic_infer;


TEST(test_layer, cat1) 
{
    int input_size = 4;
    int output_size = 2;
    int input_channels = 6;
    int output_channels = 12;

    vector<shared_ptr<Tensor<float>>> inputs;
    for (int i = 0; i < input_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(input_channels, 32, 32);
        input->Rand();
        inputs.push_back(input);
    }

    vector<shared_ptr<Tensor<float>>> outputs(output_size);
    ConcatLayer cat_layer(1);
    const auto status = cat_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->channels(), output_channels);
    }

    for (int i = 0; i < input_size / 2; ++i) {
        for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
            ASSERT_TRUE(arma::approx_equal(inputs.at(i)->at(input_channel), outputs.at(i)->at(input_channel), "absdiff", 0.01f));
        }
    }

    for (int i = input_size / 2; i < input_size; ++i) {
        for (int input_channel = input_channels; input_channel < input_channels * 2; ++input_channel) {
            ASSERT_TRUE(arma::approx_equal(inputs.at(i)->at(input_channel - input_channels), outputs.at(i - 2)->at(input_channel), "absdiff", 0.01f));
        }
    }
}
