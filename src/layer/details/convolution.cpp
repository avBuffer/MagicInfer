#include "layer/details/convolution.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "utils/tick.hpp"
#include <glog/logging.h>


namespace magic_infer 
{

ConvolutionLayer::ConvolutionLayer(uint32_t output_channel, uint32_t in_channel, uint32_t kernel_h, uint32_t kernel_w, 
    uint32_t padding_h, uint32_t padding_w, uint32_t stride_h, uint32_t stride_w, uint32_t groups, bool use_bias)
    : ParamLayer("Convolution"), padding_h_(padding_h), padding_w_(padding_w), stride_h_(stride_h), stride_w_(stride_w), 
    groups_(groups), use_bias_(use_bias) 
{
    if (groups != 1) in_channel /= groups; 
    for (uint32_t i = 0; i < output_channel; ++i) {
        shared_ptr<Tensor<float>> weight = make_shared<Tensor<float>>(in_channel, kernel_h, kernel_w);
        this->weights_.push_back(weight);
        if (use_bias_) {
            shared_ptr<Tensor<float>> bias = make_shared<Tensor<float>>(1, 1, 1);
            this->bias_.push_back(bias);
        }
    }
}


InferStatus ConvolutionLayer::Forward(const vector<shared_ptr<Tensor<float>>> &inputs, vector<shared_ptr<Tensor<float>>> &outputs) 
{
    if (inputs.empty()) {
        LOG(ERROR) << "The input feature map of convolution layer is empty";
        return InferStatus::kInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }

    if (weights_.empty()) {
        LOG(ERROR) << "Weight parameters is empty";
        return InferStatus::kInferFailedWeightParameterError;
    }

    if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
        LOG(ERROR) << "The size of the weight and bias is not adapting";
        return InferStatus::kInferFailedBiasParameterError;
    }

    const uint32_t batch_size = inputs.size();
    if (!stride_h_ || !stride_w_) {
        LOG(ERROR) << "The stride parameter is set incorrectly. It must always be greater than 0";
        return InferStatus::kInferFailedStrideParameterError;
    }

#pragma omp parallel for num_threads(batch_size)
    for (uint32_t i = 0; i < batch_size; ++i) {
        const shared_ptr<Tensor<float>> &input = inputs.at(i);
        CHECK(input != nullptr && !input->empty()) << "The input feature map of conv layer is empty";

        shared_ptr<Tensor<float>> input_;
        if (padding_h_ > 0 || padding_w_ > 0) {
            input_ = input->Clone();
            input_->Padding({padding_h_, padding_h_, padding_w_, padding_w_}, 0);
        } else {
            input_ = input;
        }

        const uint32_t input_w = input_->cols();
        const uint32_t input_h = input_->rows();
        const uint32_t input_c = input_->channels();
        const uint32_t kernel_count = this->weights_.size();

        uint32_t kernel_h = this->weights_.at(0)->rows();
        uint32_t kernel_w = this->weights_.at(0)->cols();
        CHECK(kernel_h > 0 && kernel_w > 0) << "The size of kernel size is less than zero";

        uint32_t output_h = uint32_t(floor((input_h - kernel_h) / stride_h_ + 1));
        uint32_t output_w = uint32_t(floor((input_w - kernel_w) / stride_w_ + 1));
        CHECK(output_h > 0 && output_w > 0) << "The size of the output feature map is less than zero";

        if (groups_ != 1) {
            CHECK(kernel_count % groups_ == 0);
            CHECK(input_c % groups_ == 0);
        }

        for (uint32_t k = 0; k < kernel_count; ++k) {
            const shared_ptr<Tensor<float>> &kernel = this->weights_.at(k);
            CHECK(kernel->rows() == kernel_h);
            CHECK(kernel->cols() == kernel_w);
            CHECK(kernel->channels() == input_c / groups_);
        }

        uint32_t row_len = kernel_w * kernel_h;
        uint32_t col_len = output_h * output_w;
        if (!col_len) {
            col_len = 1;
        }

        uint32_t input_c_group = input_c / groups_;
        uint32_t kernel_count_group = kernel_count / groups_;

        for (uint32_t g = 0; g < groups_; ++g) {
            vector<arma::fmat> kernel_matrix_arr(kernel_count_group);
            arma::fmat kernel_matrix_c(1, row_len * input_c_group);

            for (uint32_t k = 0; k < kernel_count_group; ++k) {
                const shared_ptr<Tensor<float>> &kernel = this->weights_.at(k + g * kernel_count_group);
                for (uint32_t ic = 0; ic < input_c_group; ++ic) {
                    memcpy(kernel_matrix_c.memptr() + row_len * ic, kernel->at(ic).memptr(), row_len * sizeof(float));
                }
                kernel_matrix_arr.at(k) = kernel_matrix_c;
            }

            arma::fmat input_matrix(input_c_group * row_len, col_len);
            for (uint32_t ic = 0; ic < input_c_group; ++ic) {
                const arma::fmat &input_channel = input_->at(ic + g * input_c_group);
                int current_col = 0;
                
                for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w_) {
                    for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h_) {
                        float *input_matrix_c_ptr = input_matrix.colptr(current_col) + ic * row_len;
                        current_col += 1;
                        
                        for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                            const float *region_ptr = input_channel.colptr(c + kw) + r;
                            memcpy(input_matrix_c_ptr, region_ptr, kernel_h * sizeof(float));
                            input_matrix_c_ptr += kernel_h;
                        }
                    }
                }
            }

            shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
            if (output_tensor == nullptr || output_tensor->empty()) {
                output_tensor = make_shared<Tensor<float>>(kernel_count, output_h, output_w);
                outputs.at(i) = output_tensor;
            }
            CHECK(output_tensor->rows() == output_h && output_tensor->cols() == output_w && output_tensor->channels() == kernel_count) 
                << "The output size of convolution is error";

            vector<arma::fmat> outputs_matrix(kernel_count_group);
#pragma omp parallel for schedule(static)
            for (uint32_t k = 0; k < kernel_count_group; ++k) {
                const arma::fmat &output = kernel_matrix_arr.at(k) * input_matrix;
                outputs_matrix.at(k) = output;
            }

            for (uint32_t k = 0; k < kernel_count_group; ++k) {
                shared_ptr<Tensor<float>> bias;
                if (!this->bias_.empty() && this->use_bias_) {
                    bias = this->bias_.at(k);
                }

                arma::fmat output = outputs_matrix.at(k);
                CHECK(output.size() == output_h * output_w);

                output.reshape(output_h, output_w);
                if (bias != nullptr) {
                    float bias_value = bias->index(0);
                    output += bias_value;
                }
                output_tensor->at(k + g * kernel_count_group) = move(output);
            }
        }
    }

    return InferStatus::kInferSuccess;
}


ParseParameterAttrStatus ConvolutionLayer::GetInstance(const shared_ptr<RuntimeOperator> &op, shared_ptr<Layer> &conv_layer) 
{
    CHECK(op != nullptr) << "Convolution operator is nullptr";
    const map<string, RuntimeParameter *> &params = op->params;

    if (params.find("in_channels") == params.end()) {
        LOG(ERROR) << "Can not find the in channel parameter";
        return ParseParameterAttrStatus::kParameterMissingInChannel;
    }

    const auto &in_channel = dynamic_cast<RuntimeParameterInt *>(params.at("in_channels"));
    if (!in_channel) {
        LOG(ERROR) << "Can not find the in channel parameter";
        return ParseParameterAttrStatus::kParameterMissingInChannel;
    }

    if (params.find("out_channels") == params.end()) {
        LOG(ERROR) << "Can not find the out channel parameter";
        return ParseParameterAttrStatus::kParameterMissingOutChannel;
    }

    const auto &out_channel = dynamic_cast<RuntimeParameterInt *>(params.at("out_channels"));
    if (!out_channel) {
        LOG(ERROR) << "Can not find the out channel parameter";
        return ParseParameterAttrStatus::kParameterMissingOutChannel;
    }

    if (params.find("padding") == params.end()) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParameterAttrStatus::kParameterMissingPadding;
    }

    const auto &padding = dynamic_cast<RuntimeParameterIntArray *>(params.at("padding"));
    if (!padding) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParameterAttrStatus::kParameterMissingPadding;
    }

    if (params.find("bias") == params.end()) {
        LOG(ERROR) << "Can not find the bias parameter";
        return ParseParameterAttrStatus::kParameterMissingUseBias;
    }

    const auto &use_bias = dynamic_cast<RuntimeParameterBool *>(params.at("bias"));
    if (!use_bias) {
        LOG(ERROR) << "Can not find the bias parameter";
        return ParseParameterAttrStatus::kParameterMissingUseBias;
    }

    if (params.find("stride") == params.end()) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParameterAttrStatus::kParameterMissingStride;
    }
    const auto &stride = dynamic_cast<RuntimeParameterIntArray *>(params.at("stride"));
    if (!stride) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParameterAttrStatus::kParameterMissingStride;
    }

    if (params.find("kernel_size") == params.end()) {
        LOG(ERROR) << "Can not find the kernel parameter";
        return ParseParameterAttrStatus::kParameterMissingKernel;
    }

    const auto &kernel = dynamic_cast<RuntimeParameterIntArray *>(params.at("kernel_size"));
    if (!kernel) {
        LOG(ERROR) << "Can not find the kernel parameter";
        return ParseParameterAttrStatus::kParameterMissingKernel;
    }

    const auto &groups = dynamic_cast<RuntimeParameterInt *>(params.at("groups"));
    if (!groups) {
        LOG(ERROR) << "Can not find the groups parameter";
        return ParseParameterAttrStatus::kParameterMissingGroups;
    }

    const uint32_t dims = 2;
    const vector<int> &kernels = kernel->value;
    const vector<int> &paddings = padding->value;
    const vector<int> &strides = stride->value;

    if (paddings.size() != dims) {
        LOG(ERROR) << "Can not find the right padding parameter";
        return ParseParameterAttrStatus::kParameterMissingPadding;
    }

    if (strides.size() != dims) {
        LOG(ERROR) << "Can not find the right stride parameter";
        return ParseParameterAttrStatus::kParameterMissingStride;
    }

    if (kernels.size() != dims) {
        LOG(ERROR) << "Can not find the right kernel size parameter";
        return ParseParameterAttrStatus::kParameterMissingKernel;
    }

    // kernel的方向是倒置的
    conv_layer = make_shared<ConvolutionLayer>(out_channel->value, in_channel->value, kernels.at(0), kernels.at(1), 
        paddings.at(0), paddings.at(1), strides.at(0), strides.at(1), groups->value, use_bias->value);

    // load weights
    const map<string, shared_ptr<RuntimeAttribute>> &attrs = op->attribute;
    if (use_bias->value) {
        if (attrs.find("bias") == attrs.end()) {
            LOG(ERROR) << "Can not find the bias attribute";
            return ParseParameterAttrStatus::kAttrMissingBias;
        }

        const auto &bias = attrs.at("bias");
        const vector<int> &bias_shape = bias->shape;
        if (bias_shape.empty() || bias_shape.at(0) != out_channel->value) {
            LOG(ERROR) << "The attribute of bias shape is wrong";
            return ParseParameterAttrStatus::kAttrMissingBias;
        }

        const vector<float> &bias_values = bias->get<float>();
        conv_layer->set_bias(bias_values);
    }

    if (attrs.find("weight") == attrs.end()) {
        LOG(ERROR) << "Can not find the weight attribute";
        return ParseParameterAttrStatus::kAttrMissingWeight;
    }

    const auto &weight = attrs.at("weight");
    const vector<int> &weight_shape = weight->shape;
    if (weight_shape.empty()) {
        LOG(ERROR) << "The attribute of weight shape is wrong";
        return ParseParameterAttrStatus::kAttrMissingWeight;
    }

    const vector<float> &weight_values = weight->get<float>();
    conv_layer->set_weights(weight_values);
    return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}


LayerRegistererWrapper kConvGetInstance("nn.Conv2d", ConvolutionLayer::GetInstance);

}
