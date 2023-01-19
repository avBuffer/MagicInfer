#ifndef MAGIC_LAYER_DETAILS_ADAPTIVE_AVGPOOLING_HPP_
#define MAGIC_LAYER_DETAILS_ADAPTIVE_AVGPOOLING_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer
{

class AdaptiveAvgPoolingLayer : public Layer 
{ 
public:
    explicit AdaptiveAvgPoolingLayer(uint32_t output_h, uint32_t output_w);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &avg_layer);

private:
    uint32_t output_h_ = 0;
    uint32_t output_w_ = 0;
};

}
#endif //MAGIC_LAYER_DETAILS_ADAPTIVE_AVGPOOLING_HPP_
