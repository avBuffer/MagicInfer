#ifndef MAGIC_LAYER_DETAILS_FLATTEN_HPP_
#define MAGIC_LAYER_DETAILS_FLATTEN_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class FlattenLayer : public Layer 
{
public:
    explicit FlattenLayer(int start_dim, int end_dim);
    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;

    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &flatten_layer);

private:
    int start_dim_ = 0;
    int end_dim_   = 0;
};

}
#endif //MAGIC_LAYER_DETAILS_FLATTEN_HPP_
