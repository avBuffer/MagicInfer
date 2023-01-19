#include "layer/details/sigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>


namespace magic_infer 
{

InferStatus SigmoidLayer::Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) 
{
    if (inputs.empty()) {
        LOG(ERROR) << "The input feature map of sigmoid layer is empty";
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
        CHECK(input == nullptr || !input->empty()) << "The input feature map of sigmoid layer is empty!";

        shared_ptr<Tensor<float>> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            output = make_shared<Tensor<float>>(input->shapes());
            outputs.at(i) = output;
        }

        CHECK (output->shapes() == input->shapes()) << "The output size of sigmoid is error";
        output->set_data(input->data());
        output->Transform([](const float value) { return 1.f / (1 + expf(-value)); });
    }
    
    return InferStatus::kInferSuccess;
}


ParseParameterAttrStatus SigmoidLayer::GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &sigmoid_layer) 
{
    CHECK(op != nullptr) << "Sigmoid operator is nullptr";
    sigmoid_layer = make_shared<SigmoidLayer>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}


LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);

}