#include "layer/details/relu.hpp"
#include "layer/abstract/layer_factory.hpp"


namespace magic_infer 
{

InferStatus ReluLayer::Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) 
{
    if (inputs.empty()) {
        LOG(ERROR) << "The input feature map of relu layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }
    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }

    const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
    for (uint32_t i = 0; i < batch_size; ++i) {
        const shared_ptr<Tensor<float>> &input = inputs.at(i);
        CHECK(input == nullptr || !input->empty()) << "The input feature map of relu layer is empty";

        shared_ptr<Tensor<float>> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            LOG(ERROR) << "The output size of relu is error";
            output = make_shared<Tensor<float>>(input->shapes());
            outputs.at(i) = output;
        }
        
        CHECK(output->shapes() == input->shapes()) << "The output size of relu is error";
        output->set_data(input->data());
        output->Transform([](float val) { return val > 0. ? val : 0.; });
    }

    return InferStatus::kInferSuccess;
}


ParseParameterAttrStatus ReluLayer::GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &relu_layer) 
{
    CHECK(op != nullptr) << "Relu operator is nullptr";
    relu_layer = make_shared<ReluLayer>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}


LayerRegistererWrapper kReluGetInstance("nn.ReLU", ReluLayer::GetInstance);

}
