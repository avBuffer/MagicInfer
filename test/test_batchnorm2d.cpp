#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"
#include "../include/layer/details/batchnorm2d.hpp"

using namespace magic_infer;


TEST(test_layer, forward_batchnorm1) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 224, 224);
    input->Rand();
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    BatchNorm2DLayer layer(3, 1e-5f, {1, 1, 1}, {0, 0, 0});
    layer.set_weights(vector<float>{0, 0, 0});
    layer.set_bias(vector<float>{1, 1, 1});

    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (const auto &output : outputs) {
        const uint32_t size = output->size();
        float mean = 0.f, var = 0.f;

        for (uint32_t i = 0; i < size; ++i) {
            mean += output->index(i);
            var  += output->index(i) * output->index(i);
        }
        ASSERT_NEAR(mean / size, 0.f, 0.01f);
        ASSERT_NEAR(var / size, 1.f, 0.01f);
    }
}


TEST(test_layer, forward_batchnorm2) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 256 * 512, 1);
    input->Rand();
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    BatchNorm2DLayer layer(1, 1e-5f, {1}, {0});
    layer.set_weights(vector<float>{0});
    layer.set_bias(vector<float>{1});

    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (const auto &output : outputs) {
        const uint32_t size = output->size();
        float mean = 0.f, var = 0.f;

        for (uint32_t i = 0; i < size; ++i) {
            mean += output->index(i);
            var  += output->index(i) * output->index(i);
        }
        ASSERT_NEAR(mean / size, 0.f, 0.01f);
        ASSERT_NEAR(var  / size, 1.f, 0.01f);
    }
}


TEST(test_layer, forward_batchnorm3) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 256, 512);
    input->Rand();
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    BatchNorm2DLayer layer(1, 1e-5f, {1}, {0});
    layer.set_weights(vector<float>{0});
    layer.set_bias(vector<float>{1});

    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (const auto &output : outputs) {
        const uint32_t size = output->size();
        float mean = 0.f, var = 0.f;
        
        for (uint32_t i = 0; i < size; ++i) {
            mean += output->index(i);
            var  += output->index(i) * output->index(i);
        }
        ASSERT_NEAR(mean / size, 0.f, 0.01f);
        ASSERT_NEAR(var  / size, 1.f, 0.01f);
    }
}


TEST(test_layer, forward_batchnorm4) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(5, 256, 512);
    input->Rand();
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    BatchNorm2DLayer layer(5, 1e-5f, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0});
    layer.set_weights(vector<float>{0, 0, 0, 0, 0});
    layer.set_bias(vector<float>{1, 1, 1, 1, 1});

    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (const auto &output : outputs) {
        const uint32_t size = output->size();
        float mean = 0.f, var = 0.f;
        
        for (uint32_t i = 0; i < size; ++i) {
            mean += output->index(i);
            var  += output->index(i) * output->index(i);
        }
        ASSERT_NEAR(mean / size, 0.f, 0.01f);
        ASSERT_NEAR(var  / size, 1.f, 0.01f);
    }
}


TEST(test_layer, forward_batchnorm5) 
{
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(7, 512, 512);
    input->Rand();
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    BatchNorm2DLayer layer(7, 1e-5f, {1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0});
    layer.set_weights(vector<float>{0, 0, 0, 0, 0, 0, 0});
    layer.set_bias(vector<float>{1, 1, 1, 1, 1, 1, 1});

    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (const auto &output : outputs) {
        const uint32_t size = output->size();
        float mean = 0.f, var = 0.f;
        
        for (uint32_t i = 0; i < size; ++i) {
            mean += output->index(i);
            var  += output->index(i) * output->index(i);
        }
        ASSERT_NEAR(mean / size, 0.f, 0.01f);
        ASSERT_NEAR(var  / size, 1.f, 0.01f);
    }
}


TEST(test_layer, forward_batchnorm6) 
{
    int batch_size = 8;
    vector<shared_ptr<Tensor<float>>> inputs;

    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 16, 16);
        input->Ones();
        inputs.push_back(input);
    }

    RuntimeGraph graph("../../weights/batchnorm/bn1.pnnx.param", "../../weights/batchnorm/bn1.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    const vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs);
    ASSERT_EQ(outputs.size(), 8);
    const auto &output_ = outputs.at(0);
    ASSERT_EQ(output_->channels(), 32);
    
    for (int i = 0; i < 32; ++i) {
        const string &path = "../../data/batchnorm/bn_" + to_string(i) + ".csv";
        const auto &output_data1 = CSVDataLoader::LoadData(path);
        const auto &output_data2 = output_->at(i);
        ASSERT_TRUE(arma::approx_equal(output_data1, output_data2, "absdiff", 1e-6));
    }
}


TEST(test_layer, forward_batchnorm7) 
{ 
    int batch_size = 8;
    vector<shared_ptr<Tensor<float>>> inputs;

    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 16, 16);
        input->Ones();
        inputs.push_back(input);
    }

    RuntimeGraph graph("../../weights/batchnorm/bn2.pnnx.param", "../../weights/batchnorm/bn2.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    
    const vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs);
    ASSERT_EQ(outputs.size(), 8);
    const auto &output_ = outputs.at(0);
    ASSERT_EQ(output_->channels(), 32);
    
    for (int i = 0; i < 32; ++i) {
        const string &path = "../../data/batchnorm/bn2_" + to_string(i) + ".csv";
        const auto &output_data1 = CSVDataLoader::LoadData(path);
        const auto &output_data2 = output_->at(i);
        ASSERT_TRUE(arma::approx_equal(output_data1, output_data2, "absdiff", 1e-4));
    }
}
