
#ifndef MAGIC_LAYER_DETAILS_BATCHNORM2D_HPP_
#define MAGIC_LAYER_DETAILS_BATCHNORM2D_HPP_

#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"

namespace magic_infer {

class BatchNorm2DLayer : public ParamLayer 
{
public:
    explicit BatchNorm2DLayer(uint32_t num_features, float eps, const vector<float> &affine_weight, const vector<float> &affine_bias);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &batch_layer);

private:
    float eps_ = 1e-5;
    vector<float> affine_weight_;
    vector<float> affine_bias_;
};

}
#endif //MAGIC_LAYER_DETAILS_BATCHNORM2D_HPP_
