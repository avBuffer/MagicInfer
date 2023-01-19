#ifndef MAGIC_LAYER_DETAILS_HARDSIGMOID_HPP_
#define MAGIC_LAYER_DETAILS_HARDSIGMOID_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class HardSigmoid : public Layer 
{
public:
    explicit HardSigmoid();

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &hardsigmoid_layer);
};

}
#endif //MAGIC_LAYER_DETAILS_HARDSIGMOID_HPP_
