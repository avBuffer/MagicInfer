#ifndef MAGIC_RUNTIME_RUNTIME_OPERAND_HPP_
#define MAGIC_RUNTIME_RUNTIME_OPERAND_HPP_

#include <vector>
#include <string>
#include <memory>

#include "utils/status_code.hpp"
#include "runtime_datatype.hpp"
#include "data/tensor.hpp"


namespace magic_infer 
{

/// 计算节点输入输出的操作数
struct RuntimeOperand 
{
    string name; /// 操作数的名称
    vector<int32_t> shapes; /// 操作数的形状
    vector<shared_ptr<Tensor<float>>> datas; /// 存储操作数
    RuntimeDataType type = RuntimeDataType::kTypeUnknown; /// 操作数的类型，一般是float
};

}
#endif //MAGIC_RUNTIME_RUNTIME_OPERAND_HPP_
