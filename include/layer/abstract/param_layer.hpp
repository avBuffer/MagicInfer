#ifndef MAGIC_LAYER_ABSTRACT_PARAM_LAYER_HPP_
#define MAGIC_LAYER_ABSTRACT_PARAM_LAYER_HPP_

#include"layer.hpp"

namespace magic_infer 
{

class ParamLayer : public Layer 
{
public:
    explicit ParamLayer(const string &layer_name);

    const vector<shared_ptr<Tensor<float>>> &weights() const override;
    const vector<shared_ptr<Tensor<float>>> &bias() const override;

    void set_weights(const vector<float> &weights) override;
    void set_bias(const vector<float> &bias) override;

    void set_weights(const vector<shared_ptr<Tensor<float>>> &weights) override;
    void set_bias(const vector<shared_ptr<Tensor<float>>> &bias) override;

protected:
    vector<shared_ptr<Tensor<float>>> weights_;
    vector<shared_ptr<Tensor<float>>> bias_;
};

}
#endif //MAGIC_LAYER_ABSTRACT_PARAM_LAYER_HPP_
