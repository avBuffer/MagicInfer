#include <glog/logging.h>
#include "layer/abstract/param_layer.hpp"


namespace magic_infer 
{

ParamLayer::ParamLayer(const string &layer_name) : Layer(layer_name) {}


const vector<shared_ptr<Tensor<float>>> &ParamLayer::weights() const 
{
    return this->weights_;
}


const vector<shared_ptr<Tensor<float>>> &ParamLayer::bias() const 
{
    return this->bias_;
}


void ParamLayer::set_weights(const vector<shared_ptr<Tensor<float>>> &weights) 
{
    this->weights_ = weights;
}


void ParamLayer::set_bias(const vector<shared_ptr<Tensor<float>>> &bias) 
{
    this->bias_ = bias;
}


void ParamLayer::set_weights(const vector<float> &weights) 
{
    const uint32_t elem_size = weights.size();
    uint32_t weight_size = 0;
    const uint32_t batch_size = this->weights_.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        weight_size += this->weights_.at(i)->size();
    }
    CHECK_EQ(weight_size, elem_size);

    const uint32_t blob_size = elem_size / batch_size;
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        const uint32_t start_offset = idx * blob_size;
        const uint32_t end_offset = start_offset + blob_size;
        const auto &sub_values = vector<float>{weights.begin() + start_offset, weights.begin() + end_offset};
        this->weights_.at(idx)->Fill(sub_values);
    }
}


void ParamLayer::set_bias(const vector<float> &bias) 
{
    const uint32_t elem_size = bias.size();
    uint32_t bias_size = 0;
    const uint32_t batch_size = this->bias_.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        bias_size += this->bias_.at(i)->size();
    }

    CHECK_EQ(bias_size, elem_size);
    const uint32_t blob_size = elem_size / batch_size;
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        const uint32_t start_offset = idx * blob_size;
        const uint32_t end_offset = start_offset + blob_size;
        const auto &sub_values = vector<float>{bias.begin() + start_offset, bias.begin() + end_offset};
        this->bias_.at(idx)->Fill(sub_values);
    }
}

}
