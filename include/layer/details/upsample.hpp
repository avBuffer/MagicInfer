
#ifndef MAGIC_LAYER_DETAILS_UPSAMPLE_HPP_
#define MAGIC_LAYER_DETAILS_UPSAMPLE_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

enum class UpSampleMode { kModeNearest = 0, }; // 目前上采样层只支持邻近采样

class UpSampleLayer : public Layer 
{
public:
    explicit UpSampleLayer(float scale_h, float scale_w, UpSampleMode mode = UpSampleMode::kModeNearest);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &upsample_layer);

private:
    float scale_h_ = 0.f;
    float scale_w_ = 0.f;
    
    UpSampleMode mode_ = UpSampleMode::kModeNearest;
};

}
#endif //MAGIC_LAYER_DETAILS_UPSAMPLE_HPP_
