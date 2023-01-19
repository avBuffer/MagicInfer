#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"

using namespace magic_infer;


TEST(test_net, forward_yolo1) 
{
    RuntimeGraph graph("../../weights/yolo/yolov5n_small.pnnx.param", "../../weights/yolo/yolov5n_small.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");

    const uint32_t batch_size = 4;
    vector<shared_ptr<Tensor<float>>> inputs;

    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 320, 320);
        input->Fill(127.f);
        inputs.push_back(input);
    }

    vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
    for (int i = 0; i < batch_size; ++i) {
        string file_path = "../../data/yolo/" + to_string(i + 1) + ".csv";
        const auto &output1 = CSVDataLoader::LoadData(file_path);
        const auto &output2 = outputs.at(i);

        ASSERT_EQ(output1.size(), output2->size());
        for (int r = 0; r < output1.n_rows; ++r) {
            for (int c = 0; c < output1.n_cols; ++c) {
                ASSERT_LE(abs(output1.at(r, c) - output2->at(0, r, c)), 0.05) << " row: " << r << " col: " << c;
            }
        }
    }
}


TEST(test_net, forward_yolo2) 
{
    RuntimeGraph graph("../../weights/yolo/yolov5n_small.pnnx.param", "../../weights/yolo/yolov5n_small.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");

    const uint32_t batch_size = 4;
    vector<shared_ptr<Tensor<float>>> inputs;

    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 320, 320);
        input->Fill(42.f);
        inputs.push_back(input);
    }

    vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
    for (int i = 0; i < batch_size; ++i) {
        string file_path = "../../data/yolo/" + to_string((i + 1) * 10 + 1) + ".csv";
        const auto &output1 = CSVDataLoader::LoadData(file_path);
        const auto &output2 = outputs.at(i);

        ASSERT_EQ(output1.size(), output2->size());
        for (int r = 0; r < output1.n_rows; ++r) {
            for (int c = 0; c < output1.n_cols; ++c) {
                ASSERT_LE(abs(output1.at(r, c) - output2->at(0, r, c)), 0.05) << " row: " << r << " col: " << c;
            }
        }
    }
}
