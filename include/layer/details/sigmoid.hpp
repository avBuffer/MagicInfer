#ifndef MAGIC_LAYER_DETAILS_SIGMOID_HPP_
#define MAGIC_LAYER_DETAILS_SIGMOID_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class SigmoidLayer : public Layer 
{
public:
    explicit SigmoidLayer(): Layer("Sigmoid") {}
    
    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &sigmoid_layer);
};

}
#endif //MAGIC_LAYER_DETAILS_SIGMOID_HPP_
