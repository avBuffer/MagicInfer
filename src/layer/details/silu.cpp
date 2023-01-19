#include "layer/details/silu.hpp"
#include "utils/tick.hpp"
#include "layer/abstract/layer_factory.hpp"

#if __SSE2__
#include <emmintrin.h>
#include "utils/sse_math.hpp"
#endif


namespace magic_infer 
{

SiLULayer::SiLULayer() : Layer("SiLU") {}


InferStatus SiLULayer::Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) 
{
    if (inputs.empty()) {
        LOG(ERROR) << "The input feature map of silu layer is empty";
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
        CHECK(input == nullptr || !input->empty()) << "The input feature map of silu layer is empty!";

        shared_ptr<Tensor<float>> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            output = make_shared<Tensor<float>>(input->shapes());
            outputs.at(i) = output;
        }

        CHECK(output->shapes() == input->shapes()) << "The output size of silu is error";
        uint32_t size = output->size();
        if (!(size % 4)) {
#if __SSE2__
            float *in_ptr = const_cast<float *>(input->RawPtr());
            float *out_ptr = const_cast<float *>(output->RawPtr());
            __m128 _one = _mm_set1_ps(1.f);
            __m128 _zero = _mm_setzero_ps();
            const uint32_t packet_size = 4;
            for (uint32_t j = 0; j + (packet_size - 1) < size; j += packet_size) {
                __m128 _p = _mm_load_ps(in_ptr);
                _p = _mm_div_ps(_p, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
                _mm_store_ps(out_ptr, _p);
                in_ptr += packet_size;
                out_ptr += packet_size;
            }
#elif
            output->set_data(input->data());
            output->Transform([](const float value) { return value / (1.f + exp(-value)); });
#endif
        } else {
            output->set_data(input->data());
            output->Transform([](const float value) { return value / (1.f + exp(-value)); });
        }
    }
    return InferStatus::kInferSuccess;
}


ParseParameterAttrStatus SiLULayer::GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &silu_layer) 
{
    CHECK(op != nullptr) << "SiLU operator is nullptr";
    silu_layer = make_shared<SiLULayer>();
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}


LayerRegistererWrapper kSiluGetInstance("nn.SiLU", SiLULayer::GetInstance);

}
