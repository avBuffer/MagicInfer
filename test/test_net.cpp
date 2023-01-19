#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"

using namespace magic_infer;


TEST(test_net, forward_resnet18) 
{
    RuntimeGraph graph("../../weights/resnet/resnet18_batch1.param", "../../weights/resnet/resnet18_batch1.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");

    int repeat_number = 2;
    for (int i = 0; i < repeat_number; ++i) {
        shared_ptr<Tensor<float>> input1 = make_shared<Tensor<float>>(3, 224, 224);
        input1->Fill(2.);

        vector<shared_ptr<Tensor<float>>> inputs;
        inputs.push_back(input1);

        vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
        ASSERT_EQ(outputs.size(), 1);

        const auto &output2 = CSVDataLoader::LoadData("../../data/resnet/23.csv");
        const auto &output1 = outputs.front()->data().slice(0);
        ASSERT_EQ(output1.size(), output2.size());
        for (uint32_t s = 0; s < output1.size(); ++s) {
            ASSERT_LE(abs(output1.at(s) - output2.at(s)), 5e-6);
        }
    }
}


TEST(test_net, forward_group_conv) 
{
    RuntimeGraph graph("../../weights/group_conv/group_conv.pnnx.param", "../../weights/group_conv/group_conv.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    shared_ptr<Tensor<float>> input1 = make_shared<Tensor<float>>(4, 16, 16);
    input1->Fill(2.f);

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);

    vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
    ASSERT_EQ(outputs.size(), 1);

    const auto &output1 = outputs.front()->data().slice(0);
    const auto &output2 = CSVDataLoader::LoadData("../../data/mobilenet/1.csv");
    ASSERT_EQ(output1.size(), output2.size());
    for (uint32_t s = 0; s < output1.size(); ++s) {
        ASSERT_LE(abs(output1.at(s) - output2.at(s)), 5e-6);
    }
}


TEST(test_net, forward_mobilenet1)
{
    RuntimeGraph graph("../../weights/mobilenet/mobile.pnnx.param", "../../weights/mobilenet/mobile.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    shared_ptr<Tensor<float>> input1 = make_shared<Tensor<float>>(3, 32, 32);
    input1->Fill(1.f);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);

    int repeat_size = 3;
    for (int i = 0; i < repeat_size; ++i) {
        vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
        ASSERT_EQ(outputs.size(), 1);

        const auto &output1 = outputs.front()->data();
        const auto &output2 = CSVDataLoader::LoadData("../../data/mobilenet/2.csv");
        ASSERT_EQ(output1.size(), output2.size());
        for (uint32_t s = 0; s < output1.size(); ++s) {
            ASSERT_LE(abs(output1.at(s) - output2.at(s)), 5e-6);
        }
    }
}


TEST(test_net, forward_mobilenet2) 
{
    RuntimeGraph graph("../../weights/mobilenet/mobile_224.pnnx.param", "../../weights/mobilenet/mobile_224.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    const uint32_t channels = 3;
    shared_ptr<Tensor<float>> input1 = make_shared<Tensor<float>>(channels, 224, 224);
    input1->Fill(1.f);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);

    int repeat_size = 3;
    for (int i = 0; i < repeat_size; ++i) {
        vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
        ASSERT_EQ(outputs.size(), 1);

        const auto &output2 = CSVDataLoader::LoadData("../../data/mobilenet/3.csv");
        const auto &output1 = outputs.front()->data();
        ASSERT_EQ(output1.size(), output2.size());

        for (uint32_t s = 0; s < output1.size(); ++s) {
            ASSERT_LE(abs(output1.at(s) - output2.at(s)), 5e-6);
        }
    }
}
