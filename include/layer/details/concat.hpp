#ifndef MAGIC_LAYER_DETAILS_CONCAT_HPP_
#define MAGIC_LAYER_DETAILS_CONCAT_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class ConcatLayer : public Layer 
{
public:
    explicit ConcatLayer(int dim);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &cat_layer);

private:
    int32_t dim_ = 0;
};

}
#endif //MAGIC_LAYER_DETAILS_CONCAT_HPP_
