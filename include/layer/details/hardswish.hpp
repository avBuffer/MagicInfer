#ifndef MAGIC_LAYER_DETAILS_HARDSWISH_HPP_
#define MAGIC_LAYER_DETAILS_HARDSWISH_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class HardSwishLayer : public Layer 
{
public:
    explicit HardSwishLayer();

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &hardswish_layer);
};

}
#endif //MAGIC_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
