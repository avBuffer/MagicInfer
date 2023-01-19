#ifndef MAGIC_RUNTIME_RUNTIME_PARAMETER_HPP_
#define MAGIC_RUNTIME_RUNTIME_PARAMETER_HPP_

#include <string>
#include <vector>
#include "utils/status_code.hpp"

using namespace std;


namespace magic_infer 
{

/**
 * 计算节点中的参数信息，参数一共可以分为如下的几类
 * 1.int
 * 2.float
 * 3.string
 * 4.bool
 * 5.int array
 * 6.string array
 * 7.float array
 */
struct RuntimeParameter 
{   
    /// 计算节点中的参数信息
    virtual ~RuntimeParameter() = default;
    explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown) : type(type) {}
    RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
};


struct RuntimeParameterInt : public RuntimeParameter 
{
    RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt) {}
    int value = 0;
};


struct RuntimeParameterFloat : public RuntimeParameter 
{
    RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat) {}
    float value = 0.f;
};


struct RuntimeParameterString : public RuntimeParameter 
{
    RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString) {}
    string value;
};


struct RuntimeParameterIntArray : public RuntimeParameter 
{
    RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {}
    vector<int> value;
};


struct RuntimeParameterFloatArray : public RuntimeParameter 
{
    RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {}
    vector<float> value;
};


struct RuntimeParameterStringArray : public RuntimeParameter 
{
    RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {}
    vector<string> value;
};


struct RuntimeParameterBool : public RuntimeParameter 
{
    RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool) {}
    bool value = false;
};

}
#endif //MAGIC_RUNTIME_RUNTIME_PARAMETER_HPP_
