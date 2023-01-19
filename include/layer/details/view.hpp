#ifndef MAGIC_LAYER_DETAILS_VIEW_HPP_
#define MAGIC_LAYER_DETAILS_VIEW_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class ViewLayer : public Layer 
{
public:
    explicit ViewLayer(const vector<int32_t> &shapes);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &view_layer);

private:
    vector<int32_t> shapes_;
};

}
#endif //MAGIC_LAYER_DETAILS_VIEW_HPP_
