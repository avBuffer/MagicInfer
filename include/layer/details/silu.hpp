#ifndef MAGIC_LAYER_DETAILS_SILU_HPP_
#define MAGIC_LAYER_DETAILS_SILU_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class SiLULayer : public Layer 
{
public:
    explicit SiLULayer();

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &silu_layer);
};

}
#endif //MAGIC_LAYER_DETAILS_SILU_HPP_
