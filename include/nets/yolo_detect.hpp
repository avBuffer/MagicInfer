#ifndef MAGIC_LAYER_DETAILS_YOLO_DETECT_HPP_
#define MAGIC_LAYER_DETAILS_YOLO_DETECT_HPP_

#include "layer/abstract/layer.hpp"
#include "layer/details/convolution.hpp"


namespace magic_infer 
{

class YoloDetectLayer : public Layer 
{
public:
    explicit YoloDetectLayer(int32_t stages, int32_t num_classes, const vector<float> &strides, const vector<arma::fmat> &anchor_grids, 
        const vector<arma::fmat> &grids, const vector<shared_ptr<ConvolutionLayer>> &conv_layers);

    InferStatus Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) override;
    static ParseParameterAttrStatus GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &yolo_detect_layer);

private:
    int32_t stages_ = 0;
    int32_t num_classes_ = 0;
    
    vector<float> strides_;
    vector<arma::fmat> anchor_grids_;
    vector<arma::fmat> grids_;
    vector<shared_ptr<ConvolutionLayer>> conv_layers_;
};

}
#endif //MAGIC_LAYER_DETAILS_YOLO_DETECT_HPP_
