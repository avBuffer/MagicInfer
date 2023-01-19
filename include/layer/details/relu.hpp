#ifndef MAGIC_LAYER_DETIALS_RELU_HPP_
#define MAGIC_LAYER_DETIALS_RELU_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class ReluLayer : public Layer 
{
public:
    ReluLayer() : Layer("Relu") {}
    
    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &relu_layer);
};

}
#endif //MAGIC_LAYER_DETIALS_RELU_HPP_
