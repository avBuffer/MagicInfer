//
// Created by fss on 22-11-13.
//

#ifndef MAGIC_LAYER_DETAILS_SOFTMAX_HPP_
#define MAGIC_LAYER_DETAILS_SOFTMAX_HPP_

#include "layer/abstract/layer.hpp"


namespace magic_infer 
{

class SoftmaxLayer : public Layer 
{
public:
    explicit SoftmaxLayer();
    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;

};

}
#endif //MAGIC_LAYER_DETAILS_SOFTMAX_HPP_
