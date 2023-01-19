#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

const vector<shared_ptr<Tensor<float>>> &Layer::weights() const 
{
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}


const vector<shared_ptr<Tensor<float>>> &Layer::bias() const 
{
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}


void Layer::set_bias(const vector<float> &bias) 
{
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}


void Layer::set_bias(const vector<shared_ptr<Tensor<float>>> &bias) 
{
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}


void Layer::set_weights(const vector<float> &weights) 
{
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}


void Layer::set_weights(const vector<shared_ptr<Tensor<float>>> &weights) 
{
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}


InferStatus Layer::Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) 
{
    LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

}