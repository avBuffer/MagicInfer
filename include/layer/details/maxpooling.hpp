#ifndef MAGIC_LAYER_DETAILS_AVGPOOLING_HPP_
#define MAGIC_LAYER_DETAILS_AVGPOOLING_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{
class MaxPoolingLayer : public Layer 
{
public:
    explicit MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
        uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &max_layer);

private:
    uint32_t padding_h_ = 0;
    uint32_t padding_w_ = 0;

    uint32_t pooling_size_h_ = 0;
    uint32_t pooling_size_w_ = 0;
    
    uint32_t stride_h_ = 1;
    uint32_t stride_w_ = 1;
};

}
#endif //MAGIC_LAYER_DETAILS_AVGPOOLING_HPP_
