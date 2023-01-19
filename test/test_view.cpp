#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "../include/layer/details/view.hpp"

using namespace magic_infer;


TEST(test_layer, forward_view1) 
{
    ViewLayer view_layer({1, 3, 32, -1});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(2, 32, 3);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(inputs.size(), 1);

    const auto &shapes = outputs.front()->shapes();
    ASSERT_EQ(shapes.at(0), 3);
    ASSERT_EQ(shapes.at(1), 32);
    ASSERT_EQ(shapes.at(2), 2);

    const auto &raw_shapes = outputs.front()->raw_shapes();
    ASSERT_EQ(raw_shapes.at(0), 3);
    ASSERT_EQ(raw_shapes.at(1), 32);
    ASSERT_EQ(raw_shapes.at(2), 2);
}


TEST(test_layer, forward_view2) 
{
    ViewLayer view_layer({1, 32, 3, 3});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 32, 3);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(inputs.size(), 1);

    const auto &shapes = outputs.front()->shapes();
    ASSERT_EQ(shapes.at(0), 32);
    ASSERT_EQ(shapes.at(1), 3);
    ASSERT_EQ(shapes.at(2), 3);

    const auto &raw_shapes = outputs.front()->raw_shapes();
    ASSERT_EQ(raw_shapes.at(0), 32);
    ASSERT_EQ(raw_shapes.at(1), 3);
    ASSERT_EQ(raw_shapes.at(2), 3);
}


TEST(test_layer, forward_view3) 
{
    ViewLayer view_layer({2, 96, 3, -1});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 32, 3);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(2);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(inputs.size(), 2);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &shapes = outputs.at(i)->shapes();
        ASSERT_EQ(shapes.at(0), 96);
        ASSERT_EQ(shapes.at(1), 3);
        ASSERT_EQ(shapes.at(2), 1);

        const auto &raw_shapes = outputs.at(i)->raw_shapes();
        ASSERT_EQ(raw_shapes.at(0), 96);
        ASSERT_EQ(raw_shapes.at(1), 3);
        ASSERT_EQ(raw_shapes.at(2), 1);
    }
}


TEST(test_layer, forward_view4) 
{
    ViewLayer view_layer({2, 3, 32, -1});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 32, 4);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(2);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(inputs.size(), 2);

    for (int i = 0; i < 2; ++i) {
        const auto &shapes = outputs.at(i)->shapes();
        ASSERT_EQ(shapes.at(0), 3);
        ASSERT_EQ(shapes.at(1), 32);
        ASSERT_EQ(shapes.at(2), 4);

        const auto &raw_shapes = outputs.at(i)->raw_shapes();
        ASSERT_EQ(raw_shapes.at(0), 3);
        ASSERT_EQ(raw_shapes.at(1), 32);
        ASSERT_EQ(raw_shapes.at(2), 4);
    }
}


TEST(test_layer, forward_view5) 
{
    ViewLayer view_layer({2, 32});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 1, 1);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(2);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(inputs.size(), 2);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &raw_shapes = outputs.at(i)->raw_shapes();
        ASSERT_EQ(raw_shapes.size(), 1);
        ASSERT_EQ(raw_shapes.at(0), 32);

        const auto &shapes = outputs.at(i)->shapes();
        ASSERT_EQ(shapes.size(), 3);
        ASSERT_EQ(shapes.at(0), 1);
        ASSERT_EQ(shapes.at(1), 32);
        ASSERT_EQ(shapes.at(2), 1);
    }
}


TEST(test_layer, forward_view6) 
{
    ViewLayer view_layer({2, 96});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 3, 1);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(2);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(inputs.size(), 2);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &raw_shapes = outputs.at(i)->raw_shapes();
        ASSERT_EQ(raw_shapes.size(), 1);
        ASSERT_EQ(raw_shapes.at(0), 96);

        const auto &shapes = outputs.at(i)->shapes();
        ASSERT_EQ(shapes.size(), 3);
        ASSERT_EQ(shapes.at(0), 1); // channels
        ASSERT_EQ(shapes.at(1), 96); // rows
        ASSERT_EQ(shapes.at(2), 1); // cols
    }
}


TEST(test_layer, forward_view7) 
{
    ViewLayer view_layer({2, 2, 3, 24});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(24, 6, 1);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(2);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(inputs.size(), 2);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &raw_shapes = outputs.at(i)->raw_shapes();
        ASSERT_EQ(raw_shapes.size(), 3);
        ASSERT_EQ(raw_shapes.at(0), 2);
        ASSERT_EQ(raw_shapes.at(1), 3);
        ASSERT_EQ(raw_shapes.at(2), 24);

        const auto &shapes = outputs.at(i)->shapes();
        ASSERT_EQ(shapes.size(), 3);
        ASSERT_EQ(shapes.at(0), 2); // channels
        ASSERT_EQ(shapes.at(1), 3); // rows
        ASSERT_EQ(shapes.at(2), 24); // cols
    }
}


TEST(test_layer, forward_view8) 
{
    ViewLayer view_layer({2, 2, 48});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 3, 1);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(2);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(inputs.size(), 2);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &shapes = outputs.at(i)->raw_shapes();
        ASSERT_EQ(shapes.size(), 2);
        ASSERT_EQ(shapes.at(0), 2);
        ASSERT_EQ(shapes.at(1), 48);

        const auto &shapes2 = outputs.at(i)->shapes();
        ASSERT_EQ(shapes2.size(), 3);
        ASSERT_EQ(shapes2.at(0), 1); // channels
        ASSERT_EQ(shapes2.at(1), 2); // rows
        ASSERT_EQ(shapes2.at(2), 48); // cols
    }
}


TEST(test_layer, forward_view9)
{
    ViewLayer view_layer({2, 3, 48});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 6, 24);
    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(2);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(inputs.size(), 2);

    for (int i = 0; i < outputs.size(); ++i) {
        const auto &shapes = outputs.at(i)->raw_shapes();
        ASSERT_EQ(shapes.size(), 2);
        ASSERT_EQ(shapes.at(0), 3);
        ASSERT_EQ(shapes.at(1), 48);

        const auto &shapes2 = outputs.at(i)->shapes();
        ASSERT_EQ(shapes2.size(), 3);
        ASSERT_EQ(shapes2.at(0), 1);    // channels
        ASSERT_EQ(shapes2.at(1), 3);    // rows
        ASSERT_EQ(shapes2.at(2), 48); // cols
    }
}


TEST(test_layer, forward_view10) 
{
    ViewLayer view_layer({1, 3, 32, 33});
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(32, 3, 33);
    uint32_t elem_size = 32 * 3 * 33;
    vector<float> values;
    srand(time(nullptr));

    for (uint32_t i = 0; i < elem_size; ++i) {
        values.push_back(float(i) + rand() % 42);
    }
    input->Fill(values);

    vector<shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = view_layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(inputs.size(), 1);

    for (int s = 0; s < outputs.size(); ++s) {
        const auto &output = outputs.at(s);
        const auto &shapes = output->shapes();
        ASSERT_EQ(shapes.at(0), 3);
        ASSERT_EQ(shapes.at(1), 32);
        ASSERT_EQ(shapes.at(2), 33);

        const auto &raw_shapes = outputs.at(s)->raw_shapes();
        ASSERT_EQ(raw_shapes.at(0), 3);
        ASSERT_EQ(raw_shapes.at(1), 32);
        ASSERT_EQ(raw_shapes.at(2), 33);
    }
}


TEST(test_layer, forward_view11) 
{
    ViewLayer layer({1, 3, 32, 32});
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 1, 3 * 32 * 32);
    const uint32_t elem_size = 3 * 32 * 32;

    vector<float> values;
    for (uint32_t i = 0; i < elem_size; ++i) {
        values.push_back(float(i));
    }

    input->Fill(values);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    const auto status = layer.Forward(inputs, outputs);
    ASSERT_EQ(status, InferStatus::kInferSuccess);

    for (int s = 0; s < outputs.size(); ++s) {
        const auto &output = outputs.at(s);
        ASSERT_EQ(output->channels(), 3);
        ASSERT_EQ(output->rows(), 32);
        ASSERT_EQ(output->cols(), 32);
    }
}


TEST(test_layer, forward_view12) 
{
    ViewLayer layer({1, 32, 3, 32});
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(1, 1, 3 * 32 * 32);
    const uint32_t elem_size = 3 * 32 * 32;
    
    vector<float> values;
    for (uint32_t i = 0; i < elem_size; ++i) {
        values.push_back(float(i));
    }
    input->Fill(values);
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    layer.Forward(inputs, outputs);

    for (int s = 0; s < outputs.size(); ++s) {
        const auto &output = outputs.at(s);
        ASSERT_EQ(output->channels(), 32);
        ASSERT_EQ(output->rows(), 3);
        ASSERT_EQ(output->cols(), 32);
    }
}


TEST(test_layer, forward_view13) 
{
    ViewLayer layer({1, 1, 32 * 3, 32});
    vector<shared_ptr<Tensor<float>>> inputs;
    shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 32, 32);
    input->Rand();
    inputs.push_back(input);

    vector<shared_ptr<Tensor<float>>> outputs(1);
    layer.Forward(inputs, outputs);

    for (int s = 0; s < outputs.size(); ++s) {
        const auto &output = outputs.at(s);
        const auto &input_ = inputs.at(s);
        const auto &slice_input = input_->data().slice(0);
        const auto &slice_output = output->data().slice(0);

        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                ASSERT_EQ(slice_input.at(i, j), slice_output.at(i, j));
            }
        }
    }
}


TEST(test_layer, forward_view14) 
{
    ViewLayer layer({2, 1, 77 * 3, 32});
    vector<shared_ptr<Tensor<float>>> inputs;

    for (int i = 0; i < 2; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 77, 32);
        input->Rand();
        inputs.push_back(input);
    }

    vector<shared_ptr<Tensor<float>>> outputs(2);
    layer.Forward(inputs, outputs);

    for (int s = 0; s < outputs.size(); ++s) {
        const auto &output = outputs.at(s);
        const auto &input_ = inputs.at(s);
        const auto &slice_input = input_->data().slice(0);
        const auto &slice_output = output->data().slice(0);

        for (int i = 0; i < 77; ++i) {
            for (int j = 0; j < 32; ++j) {
                ASSERT_EQ(slice_input.at(i, j), slice_output.at(i, j));
            }
        }
    }
}
