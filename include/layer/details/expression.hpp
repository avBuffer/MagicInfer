#ifndef MAGIC_LAYER_DETAILS_EXPRESSION_HPP_
#define MAGIC_LAYER_DETAILS_EXPRESSION_HPP_

#include "layer/abstract/layer.hpp"
#include "parser/parse_expression.hpp"


namespace magic_infer 
{

class ExpressionLayer : public Layer 
{
public:
    explicit ExpressionLayer(const string &statement);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &expression_layer);

private:
    unique_ptr<ExpressionParser> parser_;
};

}
#endif //MAGIC_LAYER_DETAILS_EXPRESSION_HPP_
