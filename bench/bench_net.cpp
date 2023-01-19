#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"

using namespace magic_infer;
const int kIterationNum = 5;


static void BM_Resnet18(benchmark::State &state) 
{
    RuntimeGraph graph("../../weights/resnet/resnet18_batch8.pnnx.param", "../../weights/resnet/resnet18_batch8.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    
    const uint32_t batch_size = 8;
    vector<shared_ptr<Tensor<float>>> inputs;
    
    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input1 = make_shared<Tensor<float>>(3, 224, 224);
        input1->Fill(1.);
        inputs.push_back(input1);
    }

    for (auto _ : state) {
        vector<shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    }
}


static void BM_Resnet18_Batch16(benchmark::State &state) 
{
    RuntimeGraph graph("../../weights/resnet/resnet18_batch16.pnnx.param", "../../weights/resnet/resnet18_batch16.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    
    const uint32_t batch_size = 16;
    vector<shared_ptr<Tensor<float>>> inputs;    
   
    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input1 = make_shared<Tensor<float>>(3, 224, 224);
        input1->Fill(1.);
        inputs.push_back(input1);
    }

    for (auto _ : state) {
        vector<shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    }
}


static void BM_MobilenetV3(benchmark::State &state) 
{
    RuntimeGraph graph("../../weights/mobilenet/mobile_batch8.pnnx.param", "../../weights/mobilenet/mobile_batch8.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    
    const uint32_t batch_size = 8;
    vector<shared_ptr<Tensor<float>>> inputs;
    
    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input1 = make_shared<Tensor<float>>(3, 224, 224);
        input1->Fill(1.);
        inputs.push_back(input1);
    }

    for (auto _ : state) {
        vector<shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    }
}


static void BM_Yolov5nano(benchmark::State &state) 
{
    RuntimeGraph graph("../../weights/yolo/yolov5n_small.pnnx.param", "../../weights/yolo/yolov5n_small.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");
    
    const uint32_t batch_size = 4;
    vector<shared_ptr<Tensor<float>>> inputs;

    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 320, 320);
        input->Ones();
        inputs.push_back(input);
    }

    for (auto _ : state) {
        vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
    }
}


static void BM_Yolov5s(benchmark::State &state) 
{
    RuntimeGraph graph("../../weights/yolo/demo/yolov5s_batch4.pnnx.param", "../../weights/yolo/demo/yolov5s_batch4.pnnx.bin");
    graph.Build("pnnx_input_0", "pnnx_output_0");

    const uint32_t batch_size = 4;
    vector<shared_ptr<Tensor<float>>> inputs;

    for (int i = 0; i < batch_size; ++i) {
        shared_ptr<Tensor<float>> input = make_shared<Tensor<float>>(3, 640, 640);
        input->Ones();
        inputs.push_back(input);
    }

    for (auto _ : state) {
        vector<shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
    }
}


BENCHMARK(BM_Resnet18)->Iterations(kIterationNum);
BENCHMARK(BM_Resnet18_Batch16)->Iterations(kIterationNum);
BENCHMARK(BM_MobilenetV3)->Iterations(kIterationNum);
BENCHMARK(BM_Yolov5nano)->Iterations(kIterationNum);
BENCHMARK(BM_Yolov5s)->Iterations(kIterationNum);
BENCHMARK_MAIN();
