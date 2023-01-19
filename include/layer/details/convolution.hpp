
#ifndef MAGIC_LAYER_DETAILS_CONVOLUTION_HPP_
#define MAGIC_LAYER_DETAILS_CONVOLUTION_HPP_

#include "layer/abstract/param_layer.hpp"


namespace magic_infer 
{

class ConvolutionLayer : public ParamLayer 
{
public:
    explicit ConvolutionLayer(uint32_t output_channel, uint32_t in_channel, uint32_t kernel_h, uint32_t kernel_w, 
        uint32_t padding_h, uint32_t padding_w, uint32_t stride_h, uint32_t stride_w, uint32_t groups, bool use_bias = true);

    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &conv_layer);
    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;

private:
    bool use_bias_ = false;
    uint32_t groups_ = 1;

    uint32_t padding_h_ = 0;
    uint32_t padding_w_ = 0;
    
    uint32_t stride_h_ = 1;
    uint32_t stride_w_ = 1;
};

}
#endif //MAGIC_LAYER_DETAILS_CONVOLUTION_HPP_
