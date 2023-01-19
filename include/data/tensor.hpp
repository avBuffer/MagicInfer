#ifndef MAGIC_DATA_TENSOR_HPP_
#define MAGIC_DATA_TENSOR_HPP_

#include <memory>
#include <vector>
#include "armadillo"

using namespace std;


namespace magic_infer 
{
template<typename T>
class Tensor {};

template<>
class Tensor<uint8_t> {};

template<>
class Tensor<float> 
{
public:
    explicit Tensor() = default;

    /**
     * 创建张量
     * @param channels 张量的通道数
     * @param rows 张量的行数
     * @param cols 张量的列数
     */
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
    explicit Tensor(const vector<uint32_t> &shapes);

    Tensor(const Tensor &tensor);

    Tensor<float> &operator=(const Tensor &tensor);

    /**
     * 返回张量的行数
     * @return 张量的行数
     */
    uint32_t rows() const;

    /**
     * 返回张量的列数
     * @return 张量的列数
     */
    uint32_t cols() const;

    /**
     * 返回张量的通道数
     * @return 张量的通道数
     */
    uint32_t channels() const;

    /**
     * 返回张量中元素的数量
     * @return 张量的元素数量
     */
    uint32_t size() const;

    /**
     * 设置张量中的具体数据
     * @param data 数据
     */
    void set_data(const arma::fcube &data);

    /**
     * 返回张量是否为空
     * @return 张量是否为空
     */
    bool empty() const;

    /**
     * 返回张量中offset位置的元素
     * @param offset 需要访问的位置
     * @return offset位置的元素
     */
    float index(uint32_t offset) const;

    /**
     * 返回张量中offset位置的元素
     * @param offset 需要访问的位置
     * @return offset位置的元素
     */
    float &index(uint32_t offset);

    /**
     * 张量的尺寸大小
     * @return 张量的尺寸大小
     */
    vector<uint32_t> shapes() const;

    /**
     * 张量的实际尺寸大小
     * @return 张量的实际尺寸大小
     */
    const vector<uint32_t> &raw_shapes() const;

    /**
     * 返回张量中的数据
     * @return 张量中的数据
     */
    arma::fcube &data();

    /**
     * 返回张量中的数据
     * @return 张量中的数据
     */
    const arma::fcube &data() const;

    /**
     * 返回张量第channel通道中的数据
     * @param channel 需要返回的通道
     * @return 返回的通道
     */
    arma::fmat &at(uint32_t channel);

    /**
     * 返回张量第channel通道中的数据
     * @param channel 需要返回的通道
     * @return 返回的通道
     */
    const arma::fmat &at(uint32_t channel) const;

    /**
     * 返回特定位置的元素
     * @param channel 通道
     * @param row 行数
     * @param col 列数
     * @return 特定位置的元素
     */
    float at(uint32_t channel, uint32_t row, uint32_t col) const;

    /**
     * 返回特定位置的元素
     * @param channel 通道
     * @param row 行数
     * @param col 列数
     * @return 特定位置的元素
     */
    float &at(uint32_t channel, uint32_t row, uint32_t col);

    /**
     * 填充张量
     * @param pads 填充张量的尺寸
     * @param padding_value 填充张量
     */
    void Padding(const vector<uint32_t> &pads, float padding_value);

    /**
     * 使用value值去初始化向量
     * @param value
     */
    void Fill(float value);

    /**
     * 使用values中的数据初始化张量
     * @param values 用来初始化张量的数据
     */
    void Fill(const vector<float> &values);

    /**
     * 以常量1初始化张量
     */
    void Ones();

    /**
     * 以随机值初始化张量
     */
    void Rand();

    /**
     * 打印张量
     */
    void Show();

    /**
     * 张量的实际尺寸大小的Reshape
     * @param shapes 张量的实际尺寸大小
     */
    void ReRawshape(const vector<uint32_t> &shapes);

    /**
     * 张量的实际尺寸大小的Reshape pytorch兼容
     * @param shapes 张量的实际尺寸大小
     */
    void ReRawView(const vector<uint32_t> &shapes);

    /**
     * 张量相加
     * @param tensor1 输入张量1
     * @param tensor2 输入张量2
     * @return 张量相加的结果
     */
    static shared_ptr<Tensor<float>> ElementAdd(
        const shared_ptr<Tensor<float>> &tensor1, const shared_ptr<Tensor<float>> &tensor2);

    /**
     * 张量相乘
     * @param tensor1 输入张量1
     * @param tensor2 输入张量2
     * @return 张量相乘的结果
     */
    static shared_ptr<Tensor<float>> ElementMultiply(
        const shared_ptr<Tensor<float>> &tensor1, const shared_ptr<Tensor<float>> &tensor2);

    /**
     * 展开张量
     */
    void Flatten();

    /**
     * 对张量中的元素进行过滤
     * @param filter 过滤函数
     */
    void Transform(const function<float(float)> &filter);

    /**
     * 返回一个深拷贝后的张量
     * @return 新的张量
     */
    shared_ptr<Tensor> Clone();

    const float *RawPtr() const;

private:
    void ReView(const vector<uint32_t> &shapes);
    
    vector<uint32_t> raw_shapes_; // 张量数据的实际尺寸大小
    arma::fcube data_; // 张量数据
};

}
#endif //MAGIC_DATA_TENSOR_HPP_
